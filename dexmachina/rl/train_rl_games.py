import os 
import math
import yaml
import torch  
import wandb 
import shutil
import pickle
import argparse
import numpy as np
from datetime import datetime 

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner 

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv 
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv


CAPTURE_MAX_START_DELAY_SECONDS = 15.0
CAPTURE_PUBLISHER_SHUTDOWN_GRACE_SECONDS = 120.0


class CaptureIsaacAlgoObserver(IsaacAlgoObserver):
    def __init__(self, capture_observer):
        super().__init__()
        self.capture_observer = capture_observer
        self.capture_frame = None

    def after_steps(self):
        if self.capture_frame is None:
            self.capture_frame = int(self.algo.frame)
        self.capture_frame += int(self.algo.curr_frames) * int(self.algo.world_size)
        if self.algo.global_rank == 0:
            frame = self.capture_frame // self.algo.num_agents
            self.capture_observer.after_print_stats(frame, self.algo.epoch_num, 0.0)


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True) 
    # make all the numpy arrays into lists, recursively
    def to_list(d):
        for k, v in d.items():
            if isinstance(v, dict):
                to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
    
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)

def main():
    parser = get_common_argparser() 
    # now add RL training args 
    parser.add_argument("--exp_name", "-exp", type=str, default="inspire", help="Experiment name.") 
    parser.add_argument("--horizon", '-ho', type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", '-ck', type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate for the agent.") 
    parser.add_argument("--wandb_project", "-wp", type=str, default="dexmachina", help="WandB project name.")
    parser.add_argument("--action_bench_experiment", type=str, default=None)
    parser.add_argument("--action_bench_catalog", type=str, default=None)
    parser.add_argument("--state-capture-dir", type=str, default=None)
    parser.add_argument("--capture-every-epochs", type=int, default=None)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument(
        "--train_root",
        type=str,
        default=None,
        help=(
            "Root directory for rl_games logs and checkpoints; "
            "logs/rl_games/<hand> is created beneath it. Defaults to the "
            "current working directory."
        ),
    )
    args = parser.parse_args()

    if args.capture_every_epochs is not None and args.capture_every_epochs <= 0:
        raise ValueError("--capture-every-epochs must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be at least one")
    if args.wandb_run_id is not None and (
        not args.wandb_run_id or any(char.isspace() for char in args.wandb_run_id)
    ):
        raise ValueError("--wandb_run_id must be nonempty and contain no whitespace")
    if (
        args.state_capture_dir is not None
        and not args.no_capture
        and not args.no_publish
    ):
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if args.action_bench_experiment is not None and args.action_mode != "hybrid":
        raise ValueError("Action-Bench training requires --action_mode hybrid")
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
    
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end 
    
    hand_prefix = str(args.hand).split("_")[0]
    exp_name = hand_prefix + "-" + args.exp_name
    exp_name += f"_{obj_name}{start}-{end}-{subject_name}-u{use_clip}_B{args.num_envs}"
    exp_name += "_"+args.action_mode
    exp_name += f"_thres{args.early_reset_threshold}"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if args.rand_init_ratio > 0:
        exp_name += f"_rand{args.rand_init_ratio}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"
        

    num_envs = args.num_envs   
    env_kwargs = get_all_env_cfg(args, device='cuda:0')
    env_kwargs['env_cfg']['use_rl_games'] = True
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    raw_env = BaseEnv(
         **env_kwargs
    )  
    env = raw_env
    if action_bench_runtime is not None:
        env = build_adapter_from_resolved_experiment(
            env,
            action_bench_runtime,
        )
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["name"] = args.hand
    
    train_root = args.train_root if args.train_root is not None else os.getcwd()
    log_root_path = os.path.join(train_root, "logs", "rl_games", args.hand)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["horizon_length"] = int(args.horizon)

    capture_schedule = None
    capture_window_seconds = None
    capture_requested = args.state_capture_dir is not None and not args.no_capture
    if capture_requested:
        from action_bench.recording import (
            DEFAULT_CAPTURE_WINDOW_SECONDS,
            make_capture_schedule,
        )

        horizon_length = int(agent_cfg["params"]["config"]["horizon_length"])
        capture_window_seconds = DEFAULT_CAPTURE_WINDOW_SECONDS
        window_steps = math.ceil(capture_window_seconds / float(raw_env.dt))
        # DexMachina epoch grid (2026-09-03): identical capture epochs for
        # every run so videos compare across arms. Grounded in the former
        # wall-clock standard (30 min x 3 h, hourly after) at the reference
        # throughput of jabs B12000 x ho32 on an L40S (~215 epochs/hour).
        capture_schedule = make_capture_schedule(
            (
                {"every_epochs": 100, "through_epoch": 600},
                {"every_epochs": 250, "through_epoch": None},
            ),
            every_epochs=args.capture_every_epochs,
        )
        print(
            "[INFO] State capture schedule "
            f"(horizon_length={horizon_length}, window_steps={window_steps}): "
            f"{capture_schedule}"
        )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args.checkpoint is not None: 
        assert os.path.exists(args.checkpoint), f"Checkpoint file not found: {args.checkpoint}"
        # agent_cfg["params"]["load_checkpoint"] = True
        # agent_cfg["params"]["load_path"] = args.checkpoint     
    
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5)) # 5 epochs per 4096 samples
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate
    
    
    env_save_kwargs = env_kwargs.copy()
    # pop the demo data and retargeted data
    env_save_kwargs.pop('demo_data')
    env_save_kwargs.pop('retarget_data')
    

    # convert agent_cfg to dict:
    wandb_cfg = agent_cfg.copy()
    wandb_cfg['env_kwargs'] = env_save_kwargs
    # also save args
    wandb_cfg['clip'] = f"{obj_name}{start}-{end}-{subject_name}-u{use_clip}"
    wandb_cfg['hand'] = args.hand
    wandb_cfg['target_ema'] = args.target_ema
    wandb_cfg['action_bench_experiment'] = args.action_bench_experiment
    wandb_cfg['action_bench_catalog'] = args.action_bench_catalog
    wandb_cfg['state_capture_publish'] = capture_requested and not args.no_publish
    
    # A tmux server booted from a launcher that touched wandb can inherit a
    # stale WANDB_SERVICE token pointing at the launcher's dead service
    # socket; wandb.init then fails with WandbServiceConnectionError. Always
    # start a fresh service in the training process.
    os.environ.pop("WANDB_SERVICE", None)
    wandb_init_kwargs = {}
    if args.wandb_run_id is not None:
        # "allow": a fresh launcher-generated id starts a new run; a reused id
        # (checkpoint resume glue) continues that run instead of erroring.
        wandb_init_kwargs.update(id=args.wandb_run_id, resume="allow")
    run = wandb.init(
        project=args.wandb_project, 
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
        **wandb_init_kwargs,
    )

    # get wandb run name and id
    run_name = run.name
    run_id = run.id
    env_save_kwargs['wandb'] = dict(
        run_name=run_name,
        run_id=run_id,
    )
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "agent.yaml"), agent_cfg) 
    # also dump as pkl file 
    pickle.dump(env_kwargs, open(os.path.join(log_root_path, exp_name, "params", "env.pkl"), "wb")) 

    state_capture = None
    capture_publisher = None
    algo_observer = IsaacAlgoObserver()
    if capture_requested and not run.disabled:
        from action_bench.benchmarks.dexmachina import CaptureEnvWrapper, build_capture
        from action_bench.recording import StateCaptureObserver

        if not args.no_publish:
            try:
                from action_bench.recording.publisher import CapturePublisher

                dexmachina_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                capture_publisher = CapturePublisher(
                    wandb_module=wandb,
                    wandb_run=run,
                    asset_roots={"dexmachina": dexmachina_root},
                )
            except Exception as error:
                print(
                    "[WARNING] Could not start capture publisher ({}: {}); "
                    "bundles will remain pending".format(
                        type(error).__name__, error
                    )
                )
        state_capture, capture_feeder = build_capture(
            raw_env,
            output_root=args.state_capture_dir,
            wandb_project=args.wandb_project,
            wandb_run_id=run.id,
            window_seconds=capture_window_seconds,
            max_start_delay_seconds=CAPTURE_MAX_START_DELAY_SECONDS,
            on_complete=(
                None if capture_publisher is None else capture_publisher.enqueue
            ),
        )
        env.env = CaptureEnvWrapper(env.env, capture_feeder, state_capture)
        algo_observer = CaptureIsaacAlgoObserver(
            StateCaptureObserver(state_capture, capture_schedule)
        )
    elif capture_requested:
        print("[INFO] State capture disabled because W&B is not active")

    # create runner from rl-games
    runner = Runner(algo_observer)
    runner.load(agent_cfg)

    # set seed of the env
    # env.seed(agent_cfg["params"]["seed"]) 
    # reset the agent and env
    runner.reset()
    # train the agent
    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint) 
    try:
        runner.run(runner_args)
    finally:
        # Best-effort rolling checkpoint on abnormal shutdown; a clean exit
        # (or a signal-triggered exit inside the train loop) already saved,
        # and a failed save must never mask the original exception.
        try:
            agent = getattr(runner, "agent", None)
            if agent is not None and agent.save_shutdown_checkpoint():
                print("[INFO] Saved shutdown rolling checkpoint")
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            if state_capture is not None:
                state_capture.close(timeout=60.0)
        finally:
            if capture_publisher is not None:
                capture_publisher.stop(
                    grace_s=CAPTURE_PUBLISHER_SHUTDOWN_GRACE_SECONDS
                )
                print(
                    "[INFO] Capture publisher failures={} dropped={}".format(
                        capture_publisher.failure_count,
                        capture_publisher.dropped_count,
                    )
                )

    # close the simulator
    exit()


if __name__ == "__main__":
    # run the main function
    main() 
    exit()
