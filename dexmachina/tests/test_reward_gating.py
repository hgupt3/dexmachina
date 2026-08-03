from types import SimpleNamespace

import pytest
import torch

from dexmachina.envs.base_env import BaseEnv


class _Robot:
    def __init__(self, bc_dist, should_be_called):
        self.bc_dist = bc_dist
        self.should_be_called = should_be_called
        self.calls = 0
        self.kpt_pos = torch.zeros(bc_dist.shape[0], 1, 3)
        self.wrist_pose = torch.zeros(bc_dist.shape[0], 7)

    def get_bc_dist(self):
        self.calls += 1
        if not self.should_be_called:
            raise AssertionError("disabled BC reward queried robot BC distance")
        return self.bc_dist


class _RewardModule:
    def __init__(self, bc_rew_weight):
        self.bc_rew_weight = bc_rew_weight
        self.bc_dist = object()

    def compute_reward(self, **kwargs):
        self.bc_dist = kwargs["bc_dist"]
        rewards = torch.arange(kwargs["actions"].shape[0], dtype=torch.float32)
        return rewards, {"task_rew": rewards.clone()}


def _minimal_env(bc_rew_weight):
    num_envs = 3
    env = BaseEnv.__new__(BaseEnv)
    env.n_objects = 1
    env.object_names = ["object"]
    env.objects = {
        "object": SimpleNamespace(
            root_pos=torch.zeros(num_envs, 3),
            root_quat=torch.zeros(num_envs, 4),
            dof_pos=torch.zeros(num_envs, 1),
        )
    }
    should_be_called = bc_rew_weight > 0.0
    env.robots = {
        "left": _Robot(torch.full((num_envs, 2), 1.0), should_be_called),
        "right": _Robot(torch.full((num_envs, 3), 2.0), should_be_called),
    }
    env.reward_module = _RewardModule(bc_rew_weight)
    env.actions = torch.zeros(num_envs, 4)
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)
    env.use_contact_reward = False
    env.observe_contact_force = False
    env.use_rl_games = True
    env.rew_buf = torch.zeros(num_envs)
    env.nan_envs = torch.zeros(num_envs, dtype=torch.bool)
    env.cumulative_task_rew = torch.zeros(num_envs)
    env.cumulative_con_rew = torch.zeros(num_envs)
    env.cumulative_imi_rew = torch.zeros(num_envs)
    env.cumulative_bc_rew = torch.zeros(num_envs)
    return env


@pytest.mark.parametrize("bc_rew_weight", [0.0, 0.2])
def test_bc_distance_is_only_computed_when_reward_uses_it(bc_rew_weight):
    env = _minimal_env(bc_rew_weight)

    BaseEnv._get_rewards(env)

    if bc_rew_weight == 0.0:
        assert env.reward_module.bc_dist is None
        assert all(robot.calls == 0 for robot in env.robots.values())
    else:
        assert all(robot.calls == 1 for robot in env.robots.values())
        assert torch.equal(
            env.reward_module.bc_dist,
            torch.tensor(
                [[1.0, 1.0, 2.0, 2.0, 2.0]] * env.actions.shape[0]
            ),
        )
