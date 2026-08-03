"""CUDA kernels for the fixed-index contact reductions.

This module is imported lazily by :mod:`dexmachina.envs.contacts`; importing
DexMachina on a host without Triton therefore continues to use the pure Torch
implementation.
"""

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in minimal installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _indexed_contact_reduce_kernel(
        n_contacts,
        pair_a,
        pair_b,
        force,
        contact_pos,
        contact_force_norm,
        a_idxs,
        b_idxs,
        force_out,
        pos_out,
        force_sum_out,
        n_envs: tl.constexpr,
        n_pairs: tl.constexpr,
        n_b: tl.constexpr,
        n_bins: tl.constexpr,
        block_bins: tl.constexpr,
        return_force: tl.constexpr,
        return_pos: tl.constexpr,
    ):
        """Reduce one environment without atomics or pair-sized workspaces."""
        env_idx = tl.program_id(0)
        bin_idx = tl.arange(0, block_bins)
        bin_mask = bin_idx < n_bins
        a_local = bin_idx // n_b
        b_local = bin_idx - a_local * n_b
        a_id = tl.load(a_idxs + a_local, mask=bin_mask, other=-1)
        b_id = tl.load(b_idxs + b_local, mask=bin_mask, other=-1)
        active_contacts = tl.load(n_contacts + env_idx)
        active_contacts = tl.maximum(0, tl.minimum(active_contacts, n_pairs))

        force_x = tl.zeros((block_bins,), dtype=tl.float32)
        force_y = tl.zeros((block_bins,), dtype=tl.float32)
        force_z = tl.zeros((block_bins,), dtype=tl.float32)
        if return_pos:
            weighted_x = tl.zeros((block_bins,), dtype=tl.float32)
            weighted_y = tl.zeros((block_bins,), dtype=tl.float32)
            weighted_z = tl.zeros((block_bins,), dtype=tl.float32)
            force_sum = tl.zeros((block_bins,), dtype=tl.float32)

        # Keep the reduction ordered by Genesis contact slot.  Besides avoiding
        # atomics, this matches the dense-mask oracle's reduction dimension.
        for pair_idx in tl.range(0, active_contacts):
            pair_offset = pair_idx * n_envs + env_idx
            observed_a = tl.load(pair_a + pair_offset)
            observed_b = tl.load(pair_b + pair_offset)
            direct = (observed_a == a_id) & (observed_b == b_id)
            flipped = (observed_a == b_id) & (observed_b == a_id)
            matched = bin_mask & (direct | flipped)
            matched_float = matched.to(tl.float32)

            vector_offset = pair_offset * 3
            if return_force:
                force_x += tl.load(force + vector_offset) * matched_float
                force_y += tl.load(force + vector_offset + 1) * matched_float
                force_z += tl.load(force + vector_offset + 2) * matched_float

            if return_pos:
                norm = tl.load(contact_force_norm + pair_offset)
                pos_x = tl.load(contact_pos + vector_offset)
                pos_y = tl.load(contact_pos + vector_offset + 1)
                pos_z = tl.load(contact_pos + vector_offset + 2)
                weighted_x += (pos_x * norm) * matched_float
                weighted_y += (pos_y * norm) * matched_float
                weighted_z += (pos_z * norm) * matched_float
                force_sum += norm * matched_float

        output_offset = (env_idx * n_bins + bin_idx) * 3
        if return_force:
            tl.store(force_out + output_offset, force_x, mask=bin_mask)
            tl.store(force_out + output_offset + 1, force_y, mask=bin_mask)
            tl.store(force_out + output_offset + 2, force_z, mask=bin_mask)

        if return_pos:
            tl.store(pos_out + output_offset, weighted_x, mask=bin_mask)
            tl.store(pos_out + output_offset + 1, weighted_y, mask=bin_mask)
            tl.store(pos_out + output_offset + 2, weighted_z, mask=bin_mask)
            tl.store(force_sum_out + env_idx * n_bins + bin_idx, force_sum, mask=bin_mask)


def is_available():
    return triton is not None


def indexed_contact_reduce(
    n_contacts,
    pair_a,
    pair_b,
    force,
    contact_pos,
    contact_force_norm,
    a_idxs,
    b_idxs,
    force_out,
    pos_out,
    force_sum_out,
    *,
    return_force,
    return_pos,
):
    """Launch the fused fp32 CUDA reduction into caller-owned outputs."""
    if triton is None:
        raise RuntimeError("Triton is not available")
    n_pairs, n_envs = pair_a.shape
    n_bins = a_idxs.numel() * b_idxs.numel()
    block_bins = triton.next_power_of_2(n_bins)
    num_warps = 1 if block_bins <= 32 else 2 if block_bins <= 64 else 4
    _indexed_contact_reduce_kernel[(n_envs,)](
        n_contacts,
        pair_a,
        pair_b,
        force,
        contact_pos,
        contact_force_norm,
        a_idxs,
        b_idxs,
        force_out,
        pos_out,
        force_sum_out,
        n_envs=n_envs,
        n_pairs=n_pairs,
        n_b=b_idxs.numel(),
        n_bins=n_bins,
        block_bins=block_bins,
        return_force=return_force,
        return_pos=return_pos,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
