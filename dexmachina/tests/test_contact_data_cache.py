"""ContactDataCache must mirror the collider fields bit-exactly.

The cache replaces per-field ``Field.to_torch()`` reads with persistent-buffer
copies and a single runtime sync; ``to_torch()`` itself is the oracle.
"""

import taichi as ti
import torch

from dexmachina.envs.contacts import ContactDataCache


class _FakeCollider:
    def __init__(self, contact_data, n_contacts):
        self.contact_data = contact_data
        self.n_contacts = n_contacts


def _build_collider(num_pairs, num_envs):
    struct_contact_data = ti.types.struct(
        geom_a=ti.i32,
        geom_b=ti.i32,
        penetration=ti.f32,
        normal=ti.types.vector(3, ti.f32),
        pos=ti.types.vector(3, ti.f32),
        friction=ti.f32,
        sol_params=ti.types.vector(7, ti.f32),
        force=ti.types.vector(3, ti.f32),
        link_a=ti.i32,
        link_b=ti.i32,
    )
    contact_data = struct_contact_data.field(
        shape=(num_pairs, num_envs), layout=ti.Layout.SOA
    )
    n_contacts = ti.field(ti.i32, shape=num_envs)
    return _FakeCollider(contact_data, n_contacts)


@ti.kernel
def _fill(contact_data: ti.template(), n_contacts: ti.template(), salt: ti.f32):
    for i, b in contact_data.force:
        contact_data.force[i, b] = ti.Vector(
            [i * 1.5 + b + salt, -b * 2.0, i * b * 0.25]
        )
        contact_data.pos[i, b] = ti.Vector([b - i * 0.5, i + salt, b * 3.0])
        contact_data.link_a[i, b] = i * 3 + b
        contact_data.link_b[i, b] = i - b
        contact_data.geom_a[i, b] = i + 100
        contact_data.geom_b[i, b] = b + 200
    for b in n_contacts:
        n_contacts[b] = b * 2


def test_cache_matches_to_torch_and_tracks_updates():
    ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)
    collider = _build_collider(num_pairs=17, num_envs=9)
    _fill(collider.contact_data, collider.n_contacts, 0.125)

    cache = ContactDataCache(
        collider, "cpu", need_geom_ids=True, need_link_ids=True, need_pos=True
    )
    cache.refresh()
    contact_data = collider.contact_data
    assert torch.equal(cache.n_contacts, collider.n_contacts.to_torch(device="cpu"))
    assert torch.equal(cache.force, contact_data.force.to_torch(device="cpu"))
    assert torch.equal(cache.pos, contact_data.pos.to_torch(device="cpu"))
    assert torch.equal(cache.link_a, contact_data.link_a.to_torch(device="cpu"))
    assert torch.equal(cache.link_b, contact_data.link_b.to_torch(device="cpu"))
    assert torch.equal(cache.geom_a, contact_data.geom_a.to_torch(device="cpu"))
    assert torch.equal(cache.geom_b, contact_data.geom_b.to_torch(device="cpu"))

    # A second refresh after mutation must track the field, not return stale data.
    _fill(collider.contact_data, collider.n_contacts, 7.5)
    cache.refresh()
    assert torch.equal(cache.force, contact_data.force.to_torch(device="cpu"))
    assert torch.equal(cache.pos, contact_data.pos.to_torch(device="cpu"))

    # Skipped fields stay unallocated so the link-only production path
    # (no geom filtering) does not pay for geom mirrors.
    lean = ContactDataCache(
        collider, "cpu", need_geom_ids=False, need_link_ids=True, need_pos=False
    )
    lean.refresh()
    assert lean.geom_a is None and lean.geom_b is None and lean.pos is None
    assert torch.equal(lean.link_a, contact_data.link_a.to_torch(device="cpu"))
