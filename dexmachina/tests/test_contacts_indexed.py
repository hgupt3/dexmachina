import pytest
import torch

from dexmachina.envs.contacts import ContactIndexFilter, index_contact_force


def _original_expanded_mask(n_contacts, pair_a, pair_b, a_idxs, b_idxs):
    """Original DexMachina implementation, kept only as the test oracle."""
    pair_a = pair_a.clone()
    pair_b = pair_b.clone()
    out_of_bounds = n_contacts.unsqueeze(0) <= torch.arange(
        pair_a.shape[0], device=pair_a.device
    ).view(-1, 1)
    pair_a[out_of_bounds] = -3
    pair_b[out_of_bounds] = -3
    mask_a = pair_a.unsqueeze(-1) == a_idxs.view(1, 1, -1)
    mask_b = pair_b.unsqueeze(-1) == b_idxs.view(1, 1, -1)
    mask = torch.logical_and(mask_a.unsqueeze(-1), mask_b.unsqueeze(-2))
    mask_a_flip = pair_a.unsqueeze(-1) == b_idxs.view(1, 1, -1)
    mask_b_flip = pair_b.unsqueeze(-1) == a_idxs.view(1, 1, -1)
    mask_flip = torch.logical_and(
        mask_a_flip.unsqueeze(-1), mask_b_flip.unsqueeze(-2)
    )
    return torch.logical_or(mask, mask_flip.transpose(-1, -2))


def _original_filter_force(force, mask):
    return (
        force.unsqueeze(-2).unsqueeze(-2) * mask.unsqueeze(-1)
    ).sum(dim=0)


def _original_group_positions(contact_pos, contact_force_norm, mask):
    contact_force_norm = (
        contact_force_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    pos_expanded = contact_pos.unsqueeze(-2).unsqueeze(-2)
    weighted_pos = pos_expanded * contact_force_norm
    weighted_sum = torch.sum(weighted_pos * mask.unsqueeze(-1), dim=0)
    force_sum = torch.sum(contact_force_norm * mask.unsqueeze(-1), dim=0)
    grouped = torch.where(
        force_sum > 0,
        weighted_sum / force_sum,
        torch.zeros_like(weighted_sum),
    )
    return grouped, (force_sum > 0).squeeze(-1)


def _original_all_positions(contact_pos, mask):
    any_b_mask = mask.any(dim=-1)
    positions = contact_pos.unsqueeze(-2) * any_b_mask.unsqueeze(-1)
    return positions.permute(1, 2, 0, 3), any_b_mask.permute(1, 2, 0)


def _case(case_name):
    generator = torch.Generator().manual_seed(8401)
    max_pairs = 12
    num_envs = 7
    a_idxs = torch.tensor([3, 5, 7, 9], dtype=torch.long)
    b_idxs = torch.tensor([20, 22, 24], dtype=torch.long)
    candidates = torch.tensor([3, 5, 7, 9, 20, 22, 24, 41, 43])
    pair_a = candidates[
        torch.randint(len(candidates), (max_pairs, num_envs), generator=generator)
    ].clone()
    pair_b = candidates[
        torch.randint(len(candidates), (max_pairs, num_envs), generator=generator)
    ].clone()
    n_contacts = torch.randint(
        0, max_pairs + 1, (num_envs,), generator=generator
    )

    if case_name == "zero_contacts":
        n_contacts.zero_()
    elif case_name == "full_overflow":
        n_contacts.fill_(max_pairs + 9)
    elif case_name == "flipped_pairs":
        n_contacts.fill_(max_pairs)
        pair_a[0] = b_idxs[0]
        pair_b[0] = a_idxs[1]
        pair_a[1] = b_idxs[2]
        pair_b[1] = a_idxs[3]
    elif case_name == "duplicate_pairs":
        n_contacts.fill_(max_pairs)
        pair_a[0:3] = a_idxs[0]
        pair_b[0:3] = b_idxs[1]
    elif case_name == "multiple_contacts_per_link":
        n_contacts.fill_(max_pairs)
        pair_a[0] = a_idxs[2]
        pair_b[0] = b_idxs[0]
        pair_a[1] = a_idxs[2]
        pair_b[1] = b_idxs[0]
        pair_a[2] = b_idxs[0]
        pair_b[2] = a_idxs[2]
        pair_a[3] = a_idxs[2]
        pair_b[3] = b_idxs[2]
    elif case_name == "empty_a_indices":
        a_idxs = torch.empty(0, dtype=torch.long)
        n_contacts.fill_(max_pairs)
    elif case_name == "empty_b_indices":
        b_idxs = torch.empty(0, dtype=torch.long)
        n_contacts.fill_(max_pairs)
    elif case_name == "empty_both_indices":
        a_idxs = torch.empty(0, dtype=torch.long)
        b_idxs = torch.empty(0, dtype=torch.long)
        n_contacts.fill_(max_pairs)
    else:
        raise AssertionError(case_name)

    force = torch.randn(max_pairs, num_envs, 3, generator=generator)
    contact_pos = torch.randn(max_pairs, num_envs, 3, generator=generator)
    return n_contacts, pair_a, pair_b, a_idxs, b_idxs, force, contact_pos


@pytest.mark.parametrize(
    "case_name",
    [
        "zero_contacts",
        "full_overflow",
        "flipped_pairs",
        "duplicate_pairs",
        "multiple_contacts_per_link",
        "empty_a_indices",
        "empty_b_indices",
        "empty_both_indices",
    ],
)
def test_index_filter_is_exactly_equal_to_original(case_name):
    n_contacts, pair_a, pair_b, a_idxs, b_idxs, force, contact_pos = _case(
        case_name
    )
    mask = _original_expanded_mask(
        n_contacts, pair_a, pair_b, a_idxs, b_idxs
    )
    expected_force = _original_filter_force(force, mask)
    force_norm = torch.norm(force, dim=-1)
    expected_grouped, expected_grouped_valid = _original_group_positions(
        contact_pos, force_norm, mask
    )
    expected_all, expected_all_valid = _original_all_positions(contact_pos, mask)

    contact_filter = ContactIndexFilter(a_idxs, b_idxs)
    contact_filter.prepare(n_contacts, pair_a, pair_b)
    actual_force = contact_filter.sum_forces(force)
    actual_grouped, actual_grouped_valid = contact_filter.group_positions(
        contact_pos, force_norm
    )
    actual_all, actual_all_valid = contact_filter.all_positions(contact_pos)

    fused_filter = ContactIndexFilter(a_idxs, b_idxs)
    fused_force, fused_grouped, fused_grouped_valid = (
        fused_filter.reduce_contacts(
            n_contacts,
            pair_a,
            pair_b,
            force,
            contact_pos=contact_pos,
        )
    )

    assert torch.equal(actual_force, expected_force)
    assert torch.equal(actual_grouped, expected_grouped)
    assert torch.equal(actual_grouped_valid, expected_grouped_valid)
    assert torch.equal(actual_all, expected_all)
    assert torch.equal(actual_all_valid, expected_all_valid)
    assert torch.equal(fused_force, expected_force)
    assert torch.equal(fused_grouped, expected_grouped)
    assert torch.equal(fused_grouped_valid, expected_grouped_valid)


def test_filter_reuses_caller_owned_outputs_exactly():
    n_contacts, pair_a, pair_b, a_idxs, b_idxs, force, contact_pos = _case(
        "multiple_contacts_per_link"
    )
    mask = _original_expanded_mask(
        n_contacts, pair_a, pair_b, a_idxs, b_idxs
    )
    contact_filter = ContactIndexFilter(a_idxs, b_idxs)
    contact_filter.prepare(n_contacts, pair_a, pair_b)

    force_out = torch.full(
        (pair_a.shape[1], len(a_idxs), len(b_idxs), 3), float("nan")
    )
    pos_out = torch.full_like(force_out, float("nan"))
    valid_out = torch.ones(force_out.shape[:-1], dtype=torch.bool)
    assert contact_filter.sum_forces(force, out=force_out) is force_out
    returned_pos, returned_valid = contact_filter.group_positions(
        contact_pos,
        torch.norm(force, dim=-1),
        out=pos_out,
        valid_out=valid_out,
    )

    expected_pos, expected_valid = _original_group_positions(
        contact_pos, torch.norm(force, dim=-1), mask
    )
    assert returned_pos is pos_out
    assert returned_valid is valid_out
    assert torch.equal(force_out, _original_filter_force(force, mask))
    assert torch.equal(pos_out, expected_pos)
    assert torch.equal(valid_out, expected_valid)

    fused_force_out = torch.full_like(force_out, float("nan"))
    fused_pos_out = torch.full_like(pos_out, float("nan"))
    fused_valid_out = torch.ones_like(valid_out)
    fused_filter = ContactIndexFilter(a_idxs, b_idxs)
    returned_force, returned_pos, returned_valid = fused_filter.reduce_contacts(
        n_contacts,
        pair_a,
        pair_b,
        force,
        contact_pos=contact_pos,
        force_out=fused_force_out,
        pos_out=fused_pos_out,
        valid_out=fused_valid_out,
    )

    assert returned_force is fused_force_out
    assert returned_pos is fused_pos_out
    assert returned_valid is fused_valid_out
    assert torch.equal(fused_force_out, _original_filter_force(force, mask))
    assert torch.equal(fused_pos_out, expected_pos)
    assert torch.equal(fused_valid_out, expected_valid)


def test_index_contact_force_compatibility_entry_point():
    n_contacts, pair_a, pair_b, a_idxs, b_idxs, force, _ = _case(
        "flipped_pairs"
    )
    mask = _original_expanded_mask(
        n_contacts, pair_a, pair_b, a_idxs, b_idxs
    )
    actual = index_contact_force(
        n_contacts, force, pair_a, pair_b, a_idxs, b_idxs
    )
    assert torch.equal(actual, _original_filter_force(force, mask))
