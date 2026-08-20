import torch

from dexmachina.envs.robot import BaseRobot


def _absolute_mode_robot(action_moving_avg, prev_targets):
    n_envs, ndof = prev_targets.shape
    robot = BaseRobot.__new__(BaseRobot)
    robot.initialized = True
    robot.action_mode = 'absolute'
    robot.action_dim = ndof
    robot.action_from_idxs = list(range(ndof))
    robot.joint_from_idxs = list(range(ndof))
    robot.joint_multipliers = torch.ones(ndof)
    robot.dof_limits = torch.tensor([[-1.0, 1.0]] * ndof)
    robot.residual_qpos = None
    robot.action_moving_avg = action_moving_avg
    robot.curr_targets = prev_targets.clone()
    robot.prev_targets = prev_targets.clone()
    return robot


def test_target_ema_blends_commanded_target_against_previous_target():
    prev = torch.tensor([[0.5, -0.5, 0.0], [0.25, 0.75, -0.25]])
    actions = torch.tensor([[-0.5, 0.5, 1.0], [0.0, -1.0, 0.5]])
    alpha = 0.2
    robot = _absolute_mode_robot(alpha, prev)

    new_targets = robot.translate_actions(actions, torch.zeros(2, dtype=torch.int32))

    commanded = -1.0 + 2.0 * (actions + 1.0) / 2.0  # absolute map on [-1, 1] limits
    expected = alpha * commanded + (1.0 - alpha) * prev
    assert torch.allclose(new_targets, expected)
    assert torch.allclose(robot.curr_targets, expected)
    assert torch.allclose(robot.prev_targets, prev)


def test_target_ema_alpha_one_is_stock_raw_target_application():
    prev = torch.tensor([[0.5, -0.5, 0.0]])
    actions = torch.tensor([[-0.5, 0.5, 1.0]])
    robot = _absolute_mode_robot(1.0, prev)

    new_targets = robot.translate_actions(actions, torch.zeros(1, dtype=torch.int32))

    commanded = -1.0 + 2.0 * (actions + 1.0) / 2.0
    assert torch.allclose(new_targets, commanded)
