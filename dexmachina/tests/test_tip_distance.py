import torch

from dexmachina.envs.math_utils import (
    closest_vertex_distances_object_frame,
    matrix_from_quat,
)


def _world_frame_oracle(local_vertices, keypoint_pos, pose):
    vertices = local_vertices.unsqueeze(0).repeat(keypoint_pos.shape[0], 1, 1)
    rotation = matrix_from_quat(pose[:, 3:7])
    world_vertices = (
        torch.einsum("nij,nkj->nki", rotation, vertices)
        + pose[:, None, :3]
    )
    return torch.cdist(keypoint_pos, world_vertices, p=2).min(dim=-1).values


def test_object_frame_tip_distances_match_world_frame_formulation():
    generator = torch.Generator().manual_seed(8402)
    num_envs, num_keypoints, num_vertices = 17, 31, 300
    local_vertices = torch.randn(
        num_vertices, 3, generator=generator, dtype=torch.float32
    )
    keypoint_pos = torch.randn(
        num_envs, num_keypoints, 3, generator=generator, dtype=torch.float32
    )
    pose = torch.randn(num_envs, 7, generator=generator, dtype=torch.float32)
    pose[:, 3:] /= torch.linalg.vector_norm(
        pose[:, 3:], dim=-1, keepdim=True
    )

    expected = _world_frame_oracle(local_vertices, keypoint_pos, pose)
    actual = closest_vertex_distances_object_frame(
        local_vertices, keypoint_pos, pose, env_chunk_size=5
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_object_frame_tip_distance_shape_and_keypoint_order_are_exact():
    local_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    )
    local_keypoints = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
        ]
    )
    pose = torch.tensor(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [-5.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    keypoint_pos = local_keypoints + pose[:, None, :3]

    actual = closest_vertex_distances_object_frame(
        local_vertices, keypoint_pos, pose, env_chunk_size=1
    )

    assert actual.shape == (2, 3)
    assert actual.dtype == torch.float32
    assert torch.equal(
        actual,
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 3.0, 2.0]]),
    )


def test_tip_distance_chunk_size_only_partitions_environments():
    generator = torch.Generator().manual_seed(8403)
    local_vertices = torch.randn(300, 3, generator=generator)
    keypoint_pos = torch.randn(19, 31, 3, generator=generator)
    pose = torch.randn(19, 7, generator=generator)
    pose[:, 3:] /= torch.linalg.vector_norm(
        pose[:, 3:], dim=-1, keepdim=True
    )

    one_environment_at_a_time = closest_vertex_distances_object_frame(
        local_vertices, keypoint_pos, pose, env_chunk_size=1
    )
    one_chunk = closest_vertex_distances_object_frame(
        local_vertices, keypoint_pos, pose, env_chunk_size=keypoint_pos.shape[0]
    )

    assert torch.equal(one_environment_at_a_time, one_chunk)
