from types import SimpleNamespace

import torch

from dexmachina.envs.base_env import BaseEnv


def _statistics_env(reward_keys):
    env = BaseEnv.__new__(BaseEnv)
    env.max_episode_length = 200
    env.reward_keys = reward_keys
    env.cumulative_task_rew = torch.tensor([1.0, 2.0, 4.0, 8.0])
    env.cumulative_con_rew = torch.tensor([3.0, 6.0, 9.0, 12.0])
    env.cumulative_imi_rew = torch.tensor([5.0, 10.0, 15.0, 20.0])
    env.cumulative_bc_rew = torch.tensor([7.0, 14.0, 21.0, 28.0])
    return env


def test_curriculum_reset_statistics_match_individual_scalar_copies_exactly():
    env = _statistics_env(['task', 'con', 'imi', 'bc'])
    env_idxs = torch.tensor([0, 2, 3])
    progressed = torch.tensor([17, 83, 199], dtype=torch.int32)

    expected_progress = progressed.float().mean().item()
    expected_rewards = {
        key: BaseEnv.normalize_episode_rew(env, rewards[env_idxs])
        for key, rewards in (
            ('task', env.cumulative_task_rew),
            ('con', env.cumulative_con_rew),
            ('imi', env.cumulative_imi_rew),
            ('bc', env.cumulative_bc_rew),
        )
    }

    progress, rewards = BaseEnv._curriculum_reset_statistics(
        env, env_idxs, progressed
    )

    assert progress == expected_progress
    assert rewards == expected_rewards


def test_curriculum_reset_statistics_only_reads_enabled_reward_keys():
    env = _statistics_env(['imi', 'task'])
    env.cumulative_con_rew = SimpleNamespace()
    env.cumulative_bc_rew = SimpleNamespace()
    env_idxs = torch.tensor([1, 3])

    progress, rewards = BaseEnv._curriculum_reset_statistics(
        env, env_idxs, torch.tensor([20, 40], dtype=torch.int32)
    )

    assert progress == 30.0
    assert list(rewards) == ['task', 'imi']
    assert rewards == {
        'task': BaseEnv.normalize_episode_rew(
            env, env.cumulative_task_rew[env_idxs]
        ),
        'imi': BaseEnv.normalize_episode_rew(
            env, env.cumulative_imi_rew[env_idxs]
        ),
    }
