import torch 
import genesis as gs 
import seaborn as sns

def get_contact_marker_cfgs(
    num_vis_contacts=10,
    radius=0.008,
    sources=['demo', 'policy'],
    obj_parts=['top', 'bottom'],
    hand_sides=['left', 'right'],
): 
    SNS_CUBE = sns.color_palette('cubehelix', 8)
    SNS_SUMMER = sns.color_palette("summer", 4)
    SNS_SPRING = sns.color_palette("spring", 4)
    SNS_RED = sns.color_palette("Reds", 4)
    SNS_BLUE = sns.color_palette("Blues", 4)
    CONTACT_MARKER_COLORS = {
        name: SNS_SUMMER[i] for i, name in enumerate([
            'demo_top_left', 'demo_top_right', 'demo_bottom_left', 'demo_bottom_right',
        ])
    }
    CONTACT_MARKER_COLORS.update({
        name: SNS_BLUE[-1] for i, name in enumerate([
            'policy_top_left', 'policy_top_right', 'policy_bottom_left', 'policy_bottom_right',
        ])
    })
    # generates all combos of sources, obj_parts, hand_sides
    contact_marker_cfgs = dict()
    for source in sources:
        for part in obj_parts:
            for side in hand_sides:
                name = f"{source}_{part}_{side}"
                color = CONTACT_MARKER_COLORS[name]
                contact_marker_cfgs[name] = {
                    'num_vis_contacts': num_vis_contacts,
                    'color': color + (0.8,),
                    'radius': radius,
                }
    return contact_marker_cfgs


class ContactDataCache:
    """Persistent torch mirrors of the Genesis collider contact buffers.

    Every taichi ``Field.to_torch()`` call allocates a fresh torch tensor and
    ends with a full taichi runtime sync, so reading the five contact fields
    the per-step query needs serializes the Python driver against the GPU five
    times per step. This cache preallocates the destination tensors once and
    refreshes them with taichi's own copy kernels, issuing a single runtime
    sync per refresh. The sync is also the required ordering point between the
    just-stepped simulation kernels and torch-side consumers.
    """

    def __init__(self, collider, device, need_geom_ids=False, need_link_ids=True, need_pos=True):
        from taichi.lang.util import to_pytorch_type

        self._collider = collider
        self._need_geom_ids = need_geom_ids
        self._need_link_ids = need_link_ids
        self._need_pos = need_pos
        device = torch.device(device)

        def _scalar_mirror(field):
            return torch.empty(field.shape, dtype=to_pytorch_type(field.dtype), device=device)

        def _vector_mirror(field):
            as_vector = field.m == 1
            shape_ext = (field.n,) if as_vector else (field.n, field.m)
            return torch.empty(
                field.shape + shape_ext, dtype=to_pytorch_type(field.dtype), device=device
            )

        contact_data = collider.contact_data
        self.n_contacts = _scalar_mirror(collider.n_contacts)
        self.force = _vector_mirror(contact_data.force)
        self.geom_a = _scalar_mirror(contact_data.geom_a) if need_geom_ids else None
        self.geom_b = _scalar_mirror(contact_data.geom_b) if need_geom_ids else None
        self.link_a = _scalar_mirror(contact_data.link_a) if need_link_ids else None
        self.link_b = _scalar_mirror(contact_data.link_b) if need_link_ids else None
        self.pos = _vector_mirror(contact_data.pos) if need_pos else None

    def refresh(self):
        """Copy the collider fields into the persistent mirrors; one sync."""
        from taichi._kernels import matrix_to_ext_arr, tensor_to_ext_arr
        from taichi.lang import runtime_ops

        contact_data = self._collider.contact_data
        tensor_to_ext_arr(self._collider.n_contacts, self.n_contacts)
        matrix_to_ext_arr(contact_data.force, self.force, True)
        if self._need_geom_ids:
            tensor_to_ext_arr(contact_data.geom_a, self.geom_a)
            tensor_to_ext_arr(contact_data.geom_b, self.geom_b)
        if self._need_link_ids:
            tensor_to_ext_arr(contact_data.link_a, self.link_a)
            tensor_to_ext_arr(contact_data.link_b, self.link_b)
        if self._need_pos:
            matrix_to_ext_arr(contact_data.pos, self.pos, True)
        runtime_ops.sync()


class ContactIndexFilter:
    """Reusable index-based filtering for one fixed pair of index sets.

    Genesis reports contacts as two global integer IDs.  The old implementation
    compared those IDs with every requested A/B combination and materialized a
    ``(max_pairs, num_envs, n_a, n_b)`` boolean tensor twice.  This object keeps
    dense global-ID-to-local-ID lookups and reuses O(max_pairs * num_envs)
    matching/scatter workspaces instead.

    Filter indices are entity link/geometry sets and are therefore expected to
    be unique non-negative global IDs.
    """

    def __init__(self, a_idxs, b_idxs):
        if not isinstance(a_idxs, torch.Tensor):
            a_idxs = torch.as_tensor(a_idxs, dtype=torch.long)
        if not isinstance(b_idxs, torch.Tensor):
            b_idxs = torch.as_tensor(b_idxs, dtype=torch.long, device=a_idxs.device)
        if a_idxs.ndim != 1 or b_idxs.ndim != 1:
            raise ValueError("a_idxs and b_idxs should be 1D tensors")
        if a_idxs.device != b_idxs.device:
            b_idxs = b_idxs.to(a_idxs.device)

        self.a_idxs = a_idxs.to(dtype=torch.long)
        self.b_idxs = b_idxs.to(dtype=torch.long)
        self.n_a = self.a_idxs.numel()
        self.n_b = self.b_idxs.numel()
        self.device = self.a_idxs.device

        if self.n_a and torch.unique(self.a_idxs).numel() != self.n_a:
            raise ValueError("a_idxs should contain unique global indices")
        if self.n_b and torch.unique(self.b_idxs).numel() != self.n_b:
            raise ValueError("b_idxs should contain unique global indices")
        if (self.n_a and self.a_idxs.min().item() < 0) or (
            self.n_b and self.b_idxs.min().item() < 0
        ):
            raise ValueError("contact filter indices should be non-negative")

        max_id = -1
        if self.n_a:
            max_id = max(max_id, self.a_idxs.max().item())
        if self.n_b:
            max_id = max(max_id, self.b_idxs.max().item())

        # IDs are stored at ID + 1.  Slot zero catches negative IDs and the
        # final slot catches IDs above max_id after clamping.
        lookup_size = max_id + 3
        self._sentinel = lookup_size - 1
        self._a_lookup = torch.full(
            (lookup_size,), -1, dtype=torch.long, device=self.device
        )
        self._b_lookup = torch.full_like(self._a_lookup, -1)
        if self.n_a:
            self._a_lookup[self.a_idxs + 1] = torch.arange(
                self.n_a, device=self.device
            )
        if self.n_b:
            self._b_lookup[self.b_idxs + 1] = torch.arange(
                self.n_b, device=self.device
            )

        self._pair_shape = None
        self._float_dtype = None
        self._prepared = False

    def _ensure_pair_workspace(self, pair_a, pair_b):
        if pair_a.shape != pair_b.shape or pair_a.ndim != 2:
            raise ValueError("pair_a and pair_b should have matching 2D shapes")
        if pair_a.device != self.device or pair_b.device != self.device:
            raise ValueError("contact pairs and filter lookups should share a device")
        if self._pair_shape == tuple(pair_a.shape):
            return

        n_pairs, n_envs = pair_a.shape
        shape = (n_pairs, n_envs)
        long_kwargs = dict(dtype=torch.long, device=self.device)
        bool_kwargs = dict(dtype=torch.bool, device=self.device)
        self._safe_a = torch.empty(shape, **long_kwargs)
        self._safe_b = torch.empty(shape, **long_kwargs)
        self._a_from_a = torch.empty(shape, **long_kwargs)
        self._b_from_b = torch.empty(shape, **long_kwargs)
        self._b_from_a = torch.empty(shape, **long_kwargs)
        self._a_from_b = torch.empty(shape, **long_kwargs)
        self._direct = torch.empty(shape, **bool_kwargs)
        self._flipped = torch.empty(shape, **bool_kwargs)
        self._temporary_mask = torch.empty(shape, **bool_kwargs)
        self._valid_contacts = torch.empty(shape, **bool_kwargs)
        self._direct_target = torch.empty(shape, **long_kwargs)
        self._flipped_target = torch.empty(shape, **long_kwargs)
        self._combined_target = torch.empty((*shape, 2), **long_kwargs)
        self._combined_mask = torch.empty((*shape, 2), **bool_kwargs)
        self._pair_ordinals = torch.arange(
            n_pairs, dtype=torch.long, device=self.device
        ).view(-1, 1)
        self._env_bin_offsets = (
            torch.arange(n_envs, dtype=torch.long, device=self.device).view(1, -1)
            * (self.n_a * self.n_b)
        )
        self._env_pos_offsets = (
            torch.arange(n_envs, dtype=torch.long, device=self.device).view(1, -1)
            * (self.n_a * n_pairs)
        )
        self._pair_shape = shape
        self._float_dtype = None
        self._position_shape = None

    def _ensure_float_workspace(self, dtype):
        if self._float_dtype == dtype:
            return
        self._combined_vector_source = torch.empty(
            (*self._pair_shape, 2, 3), dtype=dtype, device=self.device
        )
        self._weighted_position = torch.empty(
            (*self._pair_shape, 3), dtype=dtype, device=self.device
        )
        self._combined_scalar_source = torch.empty(
            (*self._pair_shape, 2), dtype=dtype, device=self.device
        )
        self._contact_norm = torch.empty(
            self._pair_shape, dtype=dtype, device=self.device
        )
        n_envs = self._pair_shape[1]
        self._force_sums = torch.empty(
            (n_envs, self.n_a, self.n_b), dtype=dtype, device=self.device
        )
        self._invalid_sums = torch.empty(
            (n_envs, self.n_a, self.n_b), dtype=torch.bool, device=self.device
        )
        self._float_dtype = dtype

    def contact_force_norm(self, force):
        """Compute force norms into a reusable pair-sized buffer."""
        self._ensure_float_workspace(force.dtype)
        torch.linalg.vector_norm(force, dim=-1, out=self._contact_norm)
        return self._contact_norm

    def prepare(self, n_contacts, pair_a, pair_b):
        """Gather local indices and validity for direct and flipped pairs."""
        self._ensure_pair_workspace(pair_a, pair_b)
        if n_contacts.shape != (pair_a.shape[1],):
            raise ValueError("n_contacts should have shape (num_envs,)")

        self._safe_a.copy_(pair_a)
        self._safe_a.add_(1).clamp_(0, self._sentinel)
        self._safe_b.copy_(pair_b)
        self._safe_b.add_(1).clamp_(0, self._sentinel)
        torch.take(self._a_lookup, self._safe_a, out=self._a_from_a)
        torch.take(self._b_lookup, self._safe_b, out=self._b_from_b)
        torch.take(self._b_lookup, self._safe_a, out=self._b_from_a)
        torch.take(self._a_lookup, self._safe_b, out=self._a_from_b)

        torch.lt(
            self._pair_ordinals,
            n_contacts.view(1, -1),
            out=self._valid_contacts,
        )
        torch.ge(self._a_from_a, 0, out=self._direct)
        torch.ge(self._b_from_b, 0, out=self._temporary_mask)
        self._direct.logical_and_(self._temporary_mask)
        self._direct.logical_and_(self._valid_contacts)

        torch.ge(self._b_from_a, 0, out=self._flipped)
        torch.ge(self._a_from_b, 0, out=self._temporary_mask)
        self._flipped.logical_and_(self._temporary_mask)
        self._flipped.logical_and_(self._valid_contacts)

        if self.n_a and self.n_b:
            self._direct_target.copy_(self._a_from_a)
            self._direct_target.mul_(self.n_b).add_(self._b_from_b)
            self._direct_target.add_(self._env_bin_offsets)
            self._direct_target.mul_(self._direct)

            self._flipped_target.copy_(self._a_from_b)
            self._flipped_target.mul_(self.n_b).add_(self._b_from_a)
            self._flipped_target.add_(self._env_bin_offsets)

            # If overlapping filter sets make both orientations select the
            # same output cell, logical OR in the old mask counted it once.
            torch.eq(
                self._direct_target,
                self._flipped_target,
                out=self._temporary_mask,
            )
            self._temporary_mask.logical_and_(self._direct)
            self._temporary_mask.logical_and_(self._flipped)
            torch.logical_not(self._temporary_mask, out=self._valid_contacts)
            self._flipped.logical_and_(self._valid_contacts)
            self._flipped_target.mul_(self._flipped)
            self._combined_target[..., 0].copy_(self._direct_target)
            self._combined_target[..., 1].copy_(self._flipped_target)
            self._combined_mask[..., 0].copy_(self._direct)
            self._combined_mask[..., 1].copy_(self._flipped)

        self._prepared = True

    def _check_output(self, output, shape, dtype):
        if output is None:
            return torch.empty(shape, dtype=dtype, device=self.device)
        if output.shape != shape or output.device != self.device or output.dtype != dtype:
            raise ValueError(
                f"output should have shape {shape}, dtype {dtype}, and device {self.device}"
            )
        return output

    def sum_forces(self, force, out=None):
        """Sum matched forces into ``(N, n_a, n_b, 3)``."""
        if not self._prepared or force.shape[:2] != self._pair_shape:
            raise RuntimeError("prepare should be called for the current contact pairs")
        shape = (self._pair_shape[1], self.n_a, self.n_b, 3)
        out = self._check_output(out, shape, force.dtype)
        out.zero_()
        if self.n_a == 0 or self.n_b == 0:
            return out

        self._ensure_float_workspace(force.dtype)
        flat_out = out.view(-1, 3)
        torch.mul(
            force.unsqueeze(2),
            self._combined_mask.unsqueeze(-1),
            out=self._combined_vector_source,
        )
        flat_out.scatter_add_(
            0,
            self._combined_target.reshape(-1, 1).expand(-1, 3),
            self._combined_vector_source.reshape(-1, 3),
        )
        return out

    def group_positions(self, contact_pos, contact_force_norm, out=None, valid_out=None):
        """Force-weight matched positions into ``(N, n_a, n_b, 3)``."""
        if not self._prepared or contact_pos.shape[:2] != self._pair_shape:
            raise RuntimeError("prepare should be called for the current contact pairs")
        shape = (self._pair_shape[1], self.n_a, self.n_b)
        out = self._check_output(out, (*shape, 3), contact_pos.dtype)
        valid_out = self._check_output(valid_out, shape, torch.bool)
        out.zero_()
        valid_out.zero_()
        if self.n_a == 0 or self.n_b == 0:
            return out, valid_out

        self._ensure_float_workspace(contact_pos.dtype)
        self._force_sums.zero_()
        flat_out = out.view(-1, 3)
        flat_sums = self._force_sums.view(-1)

        torch.mul(
            contact_pos,
            contact_force_norm.unsqueeze(-1),
            out=self._weighted_position,
        )
        torch.mul(
            self._weighted_position.unsqueeze(2),
            self._combined_mask.unsqueeze(-1),
            out=self._combined_vector_source,
        )
        flat_out.scatter_add_(
            0,
            self._combined_target.reshape(-1, 1).expand(-1, 3),
            self._combined_vector_source.reshape(-1, 3),
        )
        torch.mul(
            contact_force_norm.unsqueeze(-1),
            self._combined_mask,
            out=self._combined_scalar_source,
        )
        flat_sums.scatter_add_(
            0,
            self._combined_target.reshape(-1),
            self._combined_scalar_source.reshape(-1),
        )

        torch.gt(self._force_sums, 0, out=valid_out)
        torch.logical_not(valid_out, out=self._invalid_sums)
        self._force_sums.masked_fill_(self._invalid_sums, 1)
        out.div_(self._force_sums.unsqueeze(-1))
        out.masked_fill_(self._invalid_sums.unsqueeze(-1), 0)
        return out, valid_out

    def all_positions(self, contact_pos, out=None, valid_out=None):
        """Keep each matched contact position grouped only by its A index."""
        if not self._prepared or contact_pos.shape[:2] != self._pair_shape:
            raise RuntimeError("prepare should be called for the current contact pairs")
        n_pairs, n_envs = self._pair_shape
        shape = (n_envs, self.n_a, n_pairs)
        out = self._check_output(out, (*shape, 3), contact_pos.dtype)
        valid_out = self._check_output(valid_out, shape, torch.bool)
        out.zero_()
        valid_out.zero_()
        if self.n_a == 0 or self.n_b == 0:
            return out, valid_out

        self._ensure_float_workspace(contact_pos.dtype)
        if self._position_shape != shape:
            self._position_target = torch.empty_like(self._direct_target)
            self._position_flipped = torch.empty_like(self._flipped)
            self._position_combined_target = torch.empty_like(self._combined_target)
            self._position_combined_mask = torch.empty_like(self._combined_mask)
            self._position_counts = torch.empty(
                shape, dtype=torch.uint8, device=self.device
            )
            self._count_source = torch.empty(
                (*self._pair_shape, 2), dtype=torch.uint8, device=self.device
            )
            self._position_shape = shape

        self._position_counts.zero_()
        flat_out = out.view(-1, 3)
        flat_counts = self._position_counts.view(-1)

        self._position_target.copy_(self._a_from_a)
        self._position_target.mul_(n_pairs).add_(self._pair_ordinals)
        self._position_target.add_(self._env_pos_offsets)
        self._position_target.mul_(self._direct)
        self._position_combined_target[..., 0].copy_(self._position_target)
        self._position_combined_mask[..., 0].copy_(self._direct)

        # For this output the B dimension is reduced with any(), so two
        # orientations sharing an A cell must still contribute only once.
        self._position_flipped.copy_(self._flipped)
        torch.eq(self._a_from_a, self._a_from_b, out=self._temporary_mask)
        self._temporary_mask.logical_and_(self._direct)
        torch.logical_not(self._temporary_mask, out=self._valid_contacts)
        self._position_flipped.logical_and_(self._valid_contacts)
        self._position_target.copy_(self._a_from_b)
        self._position_target.mul_(n_pairs).add_(self._pair_ordinals)
        self._position_target.add_(self._env_pos_offsets)
        self._position_target.mul_(self._position_flipped)
        self._position_combined_target[..., 1].copy_(self._position_target)
        self._position_combined_mask[..., 1].copy_(self._position_flipped)
        torch.mul(
            contact_pos.unsqueeze(2),
            self._position_combined_mask.unsqueeze(-1),
            out=self._combined_vector_source,
        )
        flat_out.scatter_add_(
            0,
            self._position_combined_target.reshape(-1, 1).expand(-1, 3),
            self._combined_vector_source.reshape(-1, 3),
        )
        self._count_source.copy_(self._position_combined_mask)
        flat_counts.scatter_add_(
            0,
            self._position_combined_target.reshape(-1),
            self._count_source.reshape(-1),
        )
        torch.gt(self._position_counts, 0, out=valid_out)
        return out, valid_out
    
def get_filtered_contacts(
    entity_a,
    entity_b=None,
    filter_geoms_a=(),
    filter_geoms_b=(),
    filter_links_a=(),
    filter_links_b=(),
    return_geom_force=False,
    return_link_force=False,
    return_geom_pos=False,
    return_link_pos=False,
    device='cuda:0',  # NOTE: to_torch() default device is cpu
    geom_filter=None,
    link_filter=None,
    geom_force_out=None,
    link_force_out=None,
    geom_pos_out=None,
    geom_pos_valid_out=None,
    link_pos_out=None,
    link_pos_valid_out=None,
    data_cache=None,
):
    """Get filtered contact forces/positions between two entities.

    Supplying a reusable ``geom_filter``/``link_filter`` avoids rebuilding the
    lookup tables.  Supplying output tensors lets this function write directly
    into environment-owned buffers.  Supplying a ``ContactDataCache`` replaces
    the per-field ``to_torch()`` reads (one full runtime sync and one fresh
    allocation each) with persistent-buffer copies and a single sync.
    """
    contact_data = entity_a._solver.collider.contact_data
    if data_cache is not None:
        data_cache.refresh()
        n_contacts = data_cache.n_contacts
        force = data_cache.force
    else:
        n_contacts = entity_a._solver.collider.n_contacts.to_torch(device=device)
        force = contact_data.force.to_torch(device=device)
    contact_info = dict()

    if return_geom_force or return_geom_pos:
        if data_cache is not None:
            geom_a = data_cache.geom_a
            geom_b = data_cache.geom_b
        else:
            geom_a = contact_data.geom_a.to_torch(device=device)
            geom_b = contact_data.geom_b.to_torch(device=device)
        if geom_filter is None:
            geom_a_idxs = (
                torch.arange(entity_a.geom_start, entity_a.geom_end, device=force.device)
                if len(filter_geoms_a) == 0
                else torch.as_tensor(filter_geoms_a, device=force.device)
            )
            if len(filter_geoms_b) == 0:
                if entity_b is None:
                    raise ValueError("entity_b is required without filter_geoms_b")
                geom_b_idxs = torch.arange(
                    entity_b.geom_start, entity_b.geom_end, device=force.device
                )
            else:
                geom_b_idxs = torch.as_tensor(filter_geoms_b, device=force.device)
            geom_filter = ContactIndexFilter(geom_a_idxs, geom_b_idxs)
        geom_filter.prepare(n_contacts, geom_a, geom_b)

    if return_link_force or return_link_pos:
        if data_cache is not None:
            link_a = data_cache.link_a
            link_b = data_cache.link_b
        else:
            link_a = contact_data.link_a.to_torch(device=device)
            link_b = contact_data.link_b.to_torch(device=device)
        if link_filter is None:
            link_a_idxs = (
                torch.arange(entity_a.link_start, entity_a.link_end, device=force.device)
                if len(filter_links_a) == 0
                else torch.as_tensor(filter_links_a, device=force.device)
            )
            if len(filter_links_b) == 0:
                if entity_b is None:
                    raise ValueError("entity_b is required without filter_links_b")
                link_b_idxs = torch.arange(
                    entity_b.link_start, entity_b.link_end, device=force.device
                )
            else:
                link_b_idxs = torch.as_tensor(filter_links_b, device=force.device)
            link_filter = ContactIndexFilter(link_a_idxs, link_b_idxs)
        link_filter.prepare(n_contacts, link_a, link_b)

    if return_geom_force:
        contact_info['contact_force_geom_a'] = geom_filter.sum_forces(
            force, out=geom_force_out
        )
    if return_link_force:
        contact_info['contact_force_link_a'] = link_filter.sum_forces(
            force, out=link_force_out
        )

    if return_geom_pos or return_link_pos:
        if data_cache is not None:
            contact_pos = data_cache.pos
        else:
            contact_pos = contact_data.pos.to_torch(device=device)
    if return_geom_pos:
        geom_pos, geom_valid = geom_filter.all_positions(
            contact_pos, out=geom_pos_out, valid_out=geom_pos_valid_out
        )
        contact_info['contact_pos_geom_a'] = geom_pos
        contact_info['contact_pos_geom_a_valid'] = geom_valid
    if return_link_pos:
        force_norm = link_filter.contact_force_norm(force)
        link_pos, link_valid = link_filter.group_positions(
            contact_pos,
            force_norm,
            out=link_pos_out,
            valid_out=link_pos_valid_out,
        )
        contact_info['contact_pos_link_a'] = link_pos
        contact_info['contact_pos_link_a_valid'] = link_valid

    return contact_info


def index_contact_force(
    n_contacts,
    force,
    geom_a,
    geom_b,
    geom_a_idxs,
    geom_b_idxs,
    contact_filter=None,
    out=None,
):
    """Compatibility entry point for index-based force aggregation."""
    if contact_filter is None:
        contact_filter = ContactIndexFilter(geom_a_idxs, geom_b_idxs)
    contact_filter.prepare(n_contacts, geom_a, geom_b)
    return contact_filter.sum_forces(force, out=out)
