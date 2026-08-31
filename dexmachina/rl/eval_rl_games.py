import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
from datetime import datetime 

from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv

  
from collections import defaultdict
import shutil
import moviepy
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
 

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

def eval_one_episode(
    env,
    agent,
    obj_state_tensor,
    print_rew=False,
    record_video=False,
    show_reference=False,
    policy_env_idxs=None,
):
    obs = env.reset() 
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
    assert obj is not None, "No object found in the environment"
    if policy_env_idxs is None:
        policy_env_idxs = slice(None)
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    joint_target_left = left_hand.residual_qpos
    joint_target_right = right_hand.residual_qpos
    
    eval_data = defaultdict(list)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
            demo_state = obj_state_tensor[env_step]
                    
            if show_reference: # visualize the demo traj and set zero action
                obj.set_object_state(
                    root_pos=demo_state[None, :3],
                    root_quat=demo_state[None, 3:7],
                    joint_qpos=demo_state[None, 7:8],
                    env_idxs=torch.tensor([1], dtype=torch.int32, device=device),
                )
                actions[-1, :] = -1.0 
                for robot, joints in zip([left_hand, right_hand], [joint_target_left, joint_target_right]):
                    robot.set_joint_position(
                        joint_targets=joints[env_step][None],
                        env_idxs=[1],
                    ) 
            obs, rew, dones, infos = env.step(actions) 
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            # print(f"Step {env_step}: Obj pos: {obj_pos.cpu().numpy()}")
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state[policy_env_idxs].cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())

            rew_dict = uenv.rew_dict
            for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                eval_data[key].append(rew_dict[key][policy_env_idxs].cpu().numpy())
            eval_data["task_rew"].append(
                rew_dict["task_rew"][policy_env_idxs].cpu().numpy()
            )
            eval_data["reward"].append(rew[policy_env_idxs].cpu().numpy())
            eval_data["object_dropped"].append(
                (obj.root_pos[policy_env_idxs, 2] < uenv.table_height).cpu().numpy()
            )
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0 
        
        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes 
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    frames = uenv.get_recorded_frames()
    return frames, eval_data


def set_and_verify_eval_object_gains(env, gains):
    kp, kv, force_range = gains
    assert not env.use_curriculum, "Fixed eval object gains require curriculum to be disabled"
    assert len(env.objects) > 0, "No object found in the environment"

    for name, obj in env.objects.items():
        assert obj.actuated, f"Object {name} was not built as actuated"
        obj.set_joint_gains(kp, kv, force_range=force_range)

        dof_idxs = list(range(7))
        actual_kp = obj.entity.get_dofs_kp(dof_idxs)
        actual_kv = obj.entity.get_dofs_kv(dof_idxs)
        actual_lower, actual_upper = obj.entity.get_dofs_force_range(dof_idxs)
        expected_kp = torch.full_like(actual_kp, kp)
        expected_kv = torch.full_like(actual_kv, kv)
        expected_lower = torch.full_like(actual_lower, -force_range)
        expected_upper = torch.full_like(actual_upper, force_range)
        assert torch.allclose(actual_kp, expected_kp)
        assert torch.allclose(actual_kv, expected_kv)
        assert torch.allclose(actual_lower, expected_lower)
        assert torch.allclose(actual_upper, expected_upper)

        print(f"Object {name!r} DOF kp after setting: {actual_kp[0].cpu().tolist()}")
        print(f"Object {name!r} DOF kv after setting: {actual_kv[0].cpu().tolist()}")
        print(
            f"Object {name!r} DOF force range after setting: "
            f"lower={actual_lower[0].cpu().tolist()}, "
            f"upper={actual_upper[0].cpu().tolist()}"
        )
        print(
            f"Verified fixed gains across {actual_kp.shape[0]} envs and "
            f"{actual_kp.shape[1]} object DOFs"
        )


def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint
    parser.add_argument("--action_bench_experiment", type=str, default=None)
    parser.add_argument("--action_bench_catalog", type=str, default=None)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument(
        "--eval_object_gains",
        type=float,
        nargs=3,
        metavar=("KP", "KV", "FORCE_RANGE"),
        default=None,
    )
    
    args = parser.parse_args()

    action_bench_runtime = None
    if args.action_bench_experiment is not None:
        from action_bench.benchmarks.dexmachina import (
            build_adapter_from_resolved_experiment,
            resolve_experiment_from_files,
        )
        action_bench_runtime = resolve_experiment_from_files(
            experiment_path=args.action_bench_experiment,
            catalog_path=args.action_bench_catalog,
        )

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = args.eval_dir
    if ckpt_data_folder is None:
        ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    os.makedirs(ckpt_data_folder, exist_ok=True) 

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    # load to pkl
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        # env_kwargs['camera_res'] = (2048, 2048)
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        ) 
    for name, cfg in env_kwargs['object_cfgs'].items():
        if args.eval_object_gains is None:
            print("Setting eval time obj gains to 0.0")
            cfg['actuated'] = False
        else:
            kp, kv, force_range = args.eval_object_gains
            print(
                f"Building object {name!r} as actuated with fixed eval gains: "
                f"kp={kp}, kv={kv}, force_range={force_range}"
            )
            cfg['actuated'] = True
            cfg['kp'] = kp
            cfg['kv'] = kv
            cfg['force_range'] = force_range
            env_kwargs['env_cfg']['scene_kwargs']['batch_dofs_info'] = True

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    if args.eval_object_gains is None:
        print("Removing any curriculum config during eval")
        env_kwargs.pop("curriculum_cfg")
    else:
        print("Keeping saved curriculum config for actuated-object construction only")
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    if args.eval_object_gains is not None:
        print("Disabling object-gain curriculum before eval rollout")
        env.use_curriculum = False
        env.curriculum = None
        env.curr_cfg = dict()
        set_and_verify_eval_object_gains(env, args.eval_object_gains)
    if action_bench_runtime is not None:
        env = build_adapter_from_resolved_experiment(
            env,
            action_bench_runtime,
        )
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data)

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped 
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint) 
    agent.restore(resume_path)
    agent.reset()

    policy_env_idxs = None
    if args.eval_object_gains is not None:
        policy_env_idxs = torch.as_tensor(
            uenv._step_env_idxs, dtype=torch.long, device=device
        )
        print(
            f"Recording {policy_env_idxs.numel()} policy-controlled eval envs "
            f"out of {uenv.num_envs} built envs"
        )

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env,
            agent,
            obj_state_tensor,
            args.print_rew,
            args.record_video,
            args.show_reference,
            policy_env_idxs,
            )
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        print(f"Saved eval data to {ckpt_eval_fname}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item() 
        if args.record_video: 
            # save video with moviepy
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(frames, fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")
    
    print("Done evaluating")


if __name__ == '__main__':
    main()
