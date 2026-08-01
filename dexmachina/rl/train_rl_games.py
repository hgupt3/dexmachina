import os 
import json
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


CAPTURE_WINDOW_SECONDS = 30.0
CAPTURE_MAX_START_DELAY_SECONDS = 15.0
CAPTURE_PUBLISHER_SHUTDOWN_GRACE_SECONDS = 120.0


def _round_to_multiple(value, multiple):
    return max(multiple, ((value + multiple // 2) // multiple) * multiple)


def _default_capture_schedule(window_steps, horizon_length):
    window_epochs = math.ceil(window_steps / horizon_length)
    return [
        {
            "every_epochs": _round_to_multiple(2 * window_epochs, 25),
            "through_epoch": 2000,
        },
        {
            "every_epochs": _round_to_multiple(6 * window_epochs, 50),
            "through_epoch": None,
        },
    ]


def _parse_capture_schedule(raw):
    try:
        schedule = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("--capture-schedule must be valid JSON") from error
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("--capture-schedule must be a nonempty JSON list")
    previous_through = None
    for index, entry in enumerate(schedule):
        if not isinstance(entry, dict) or set(entry) != {
            "every_epochs",
            "through_epoch",
        }:
            raise ValueError(
                "capture schedule entries require exactly every_epochs and "
                "through_epoch"
            )
        every_epochs = entry["every_epochs"]
        if (
            isinstance(every_epochs, bool)
            or not isinstance(every_epochs, int)
            or every_epochs < 1
        ):
            raise ValueError("capture schedule every_epochs must be at least one")
        through_epoch = entry["through_epoch"]
        if through_epoch is None:
            if index != len(schedule) - 1:
                raise ValueError(
                    "only the last capture schedule entry may be open-ended"
                )
            continue
        if (
            isinstance(through_epoch, bool)
            or not isinstance(through_epoch, int)
            or through_epoch < 1
        ):
            raise ValueError(
                "capture schedule through_epoch must be a positive integer or null"
            )
        if previous_through is not None and through_epoch <= previous_through:
            raise ValueError(
                "capture schedule through_epoch values must be strictly increasing"
            )
        previous_through = through_epoch
    return schedule


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
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    parser.add_argument("--action_bench_experiment", type=str, default=None)
    parser.add_argument("--action_bench_catalog", type=str, default=None)
    parser.add_argument("--state-capture-dir", type=str, default=None)
    parser.add_argument("--capture-every", type=int, default=None)
    parser.add_argument("--capture-schedule", type=str, default=None)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    args = parser.parse_args()

    if args.capture_every is not None and args.capture_every < 1:
        raise ValueError("--capture-every must be at least one")
    if args.capture_every is not None and args.capture_schedule is not None:
        raise ValueError(
            "--capture-every and --capture-schedule are mutually exclusive"
        )
    capture_schedule_override = (
        None
        if args.capture_schedule is None
        else _parse_capture_schedule(args.capture_schedule)
    )
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
    
    log_root_path = os.path.join("logs", "rl_games", args.hand)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)
    agent_cfg["params"]["config"]["horizon_length"] = int(args.horizon)

    capture_schedule = None
    capture_requested = args.state_capture_dir is not None and not args.no_capture
    if capture_requested:
        horizon_length = int(agent_cfg["params"]["config"]["horizon_length"])
        window_steps = math.ceil(CAPTURE_WINDOW_SECONDS / float(raw_env.dt))
        capture_schedule = (
            capture_schedule_override
            if capture_schedule_override is not None
            else (
                [{"every_epochs": args.capture_every, "through_epoch": None}]
                if args.capture_every is not None
                else _default_capture_schedule(window_steps, horizon_length)
            )
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
    wandb_cfg['action_bench_experiment'] = args.action_bench_experiment
    wandb_cfg['action_bench_catalog'] = args.action_bench_catalog
    wandb_cfg['state_capture_publish'] = capture_requested and not args.no_publish
    
    wandb_init_kwargs = {}
    if args.wandb_run_id is not None:
        wandb_init_kwargs.update(id=args.wandb_run_id, resume="never")
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
            window_seconds=CAPTURE_WINDOW_SECONDS,
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
