import numpy as np
import pytest
import torch

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.curriculum import Curriculum, get_curriculum_cfg


NUM_ENVS = 4
NDOF = 3
OBJ_NDOF = 7


class FakeEntity:
    def __init__(self, ndof, actuated=False):
        self.dof_pos = torch.rand(NUM_ENVS, ndof)
        self.dof_vel = torch.rand(NUM_ENVS, ndof)
        self.root_pos = torch.rand(NUM_ENVS, 3)
        self.root_quat = torch.rand(NUM_ENVS, 4)
        if actuated:
            self.kp = torch.rand(NUM_ENVS, ndof) * 80.0
            self.kv = torch.rand(NUM_ENVS, ndof) * 5.0
            self.force_lower = -torch.rand(NUM_ENVS, ndof) * 50.0
            self.force_upper = torch.rand(NUM_ENVS, ndof) * 50.0

    def get_pos(self):
        return self.root_pos.clone()

    def get_quat(self):
        return self.root_quat.clone()

    def set_pos(self, pos, zero_velocity=True):
        self.root_pos = pos.clone()

    def set_quat(self, quat, zero_velocity=True):
        self.root_quat = quat.clone()

    def get_dofs_position(self, dofs_idx_local=None):
        if dofs_idx_local is not None:
            return self.dof_pos[:, dofs_idx_local].clone()
        return self.dof_pos.clone()

    def get_dofs_velocity(self):
        return self.dof_vel.clone()

    def set_dofs_position(self, position, dofs_idx_local=None, zero_velocity=True):
        if dofs_idx_local is not None:
            self.dof_pos[:, dofs_idx_local] = position.clone()
        else:
            self.dof_pos = position.clone()
        if zero_velocity:
            self.dof_vel = torch.zeros_like(self.dof_vel)

    def set_dofs_velocity(self, velocity):
        self.dof_vel = velocity.clone()

    def get_dofs_kp(self):
        return self.kp.clone()

    def get_dofs_kv(self):
        return self.kv.clone()

    def get_dofs_force_range(self):
        return self.force_lower.clone(), self.force_upper.clone()

    def set_dofs_kp(self, kp):
        self.kp = kp.clone()

    def set_dofs_kv(self, kv):
        self.kv = kv.clone()

    def set_dofs_force_range(self, lower, upper):
        self.force_lower = lower.clone()
        self.force_upper = upper.clone()


class FakeBody:
    def __init__(self, ndof, actuated=False):
        self.entity = FakeEntity(ndof, actuated=actuated)
        self.dof_pos = torch.zeros(NUM_ENVS, ndof)
        self.dof_vel = torch.zeros(NUM_ENVS, ndof)
        self.root_pos = torch.zeros(NUM_ENVS, 3)
        self.root_quat = torch.zeros(NUM_ENVS, 4)
        self.dof_idxs = [ndof - 1]
        self.curr_targets = torch.rand(NUM_ENVS, ndof)
        self.prev_targets = torch.rand(NUM_ENVS, ndof)
        self.episode_length_buf = torch.randint(
            0, 100, (NUM_ENVS,), dtype=torch.int32
        )
        self.actuated = actuated

    def update_value_buffers(self):
        pass


def _fake_env():
    env = BaseEnv.__new__(BaseEnv)
    env.warm_env_state = True
    env.is_eval = False
    env.num_envs = NUM_ENVS
    env.device = torch.device('cpu')
    env._pending_env_state = None
    env._warm_resume_report_pending = False
    env.episode_length_buf = torch.randint(0, 100, (NUM_ENVS,), dtype=torch.int32)
    env.episode_start_buf = torch.zeros(NUM_ENVS, dtype=torch.int32)
    env.max_achieved_length = 57
    env.cumulative_task_rew = torch.rand(NUM_ENVS)
    env.cumulative_con_rew = torch.rand(NUM_ENVS)
    env.cumulative_imi_rew = torch.rand(NUM_ENVS)
    env.cumulative_bc_rew = torch.rand(NUM_ENVS)
    env.actions = torch.rand(NUM_ENVS, 2 * NDOF)
    env.last_actions = torch.rand(NUM_ENVS, 2 * NDOF)
    env.nan_envs = torch.zeros(NUM_ENVS, dtype=torch.bool)
    env.reset_buf = torch.zeros(NUM_ENVS, dtype=torch.bool)
    env.reset_terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
    env.robots = {'left': FakeBody(NDOF), 'right': FakeBody(NDOF)}
    env.objects = {'box': FakeBody(OBJ_NDOF, actuated=True)}
    env.use_curriculum = True
    env.curriculum = Curriculum(
        get_curriculum_cfg(),
        task_object=None,
        reward_keys=['task', 'imi'],
        num_envs=NUM_ENVS,
        achieved_length=0,
        max_episode_length=200,
    )
    # Evolve the curriculum away from its initial state.
    for value in (0.4, 0.5, 0.6):
        env.curriculum.update_progress({'task': value, 'imi': value}, 150)
    env.curriculum.gain_history.append(
        dict(gains=dict(env.curriculum.curr_gains), lower=dict(env.curriculum.curr_gains_lower))
    )
    env.curriculum.curr_gains = {k: v * 0.9 for k, v in env.curriculum.curr_gains.items()}
    env.curriculum.num_epoch_since_last_decay = 17
    env.curriculum.num_epoch_since_zero = 3
    return env


def _assert_same(a, b, path=""):
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{path}: array meta"
        assert np.array_equal(a, b), f"{path}: array values differ"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_env_state_roundtrip_is_exact():
    env = _fake_env()
    first = env.get_env_state()
    assert first is not None and first['version'] == 1

    # Perturb everything the state covers, then restore and re-capture.
    env.episode_length_buf.zero_()
    env.cumulative_task_rew.zero_()
    env.actions.zero_()
    for body in (*env.robots.values(), *env.objects.values()):
        body.curr_targets.zero_()
        body.entity.dof_pos.zero_()
        body.entity.dof_vel.zero_()
    env.objects['box'].entity.kp.zero_()
    env.curriculum.curr_gains = {k: 0.0 for k in env.curriculum.curr_gains}
    env.curriculum.rew_deques['task'].clear()
    np.random.rand(5)
    torch.rand(5)

    env.set_env_state(first)
    assert env._pending_env_state is first
    env._pending_env_state = None
    env._apply_env_state(first)
    second = env.get_env_state()
    _assert_same(first, second)
    assert env._warm_resume_report_pending


def test_env_state_missing_or_mismatched_is_handled():
    env = _fake_env()
    env.set_env_state(None)  # old checkpoint: warn and keep current behavior
    assert env._pending_env_state is None
    state = env.get_env_state()
    state['num_envs'] = NUM_ENVS + 1
    with pytest.raises(ValueError, match="envs"):
        env.set_env_state(state)
    env.is_eval = True  # players restore checkpoints too; eval envs ignore
    env.set_env_state(state)
    assert env._pending_env_state is None
    env.is_eval = False
    env.warm_env_state = False
    assert env.get_env_state() is None
