"""Run one bootstrap and three controlled MJLab PPGA descriptor experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DESCRIPTOR_MODES = (
    "contact_azimuth_height",
    "approach_orientation",
    "contact_transport",
)


def ppga_command(args, mode: str, checkpoint: Path,
                 scheduler_checkpoint: Path | None = None) -> list[str]:
    initialization = (
        f"--load_scheduler_from_cp={scheduler_checkpoint}"
        if scheduler_checkpoint is not None
        else f"--initial_actor_checkpoint={checkpoint}"
    )
    return [
        sys.executable, "-u", "-m", "ppga.algorithm.train_ppga_isaac",
        "--env_type=mjlab", "--env_name=lift_cube",
        initialization,
        "--mjlab_fixed_goal=False", "--mjlab_disable_curriculum=True",
        "--mjlab_command_resampling_time=40.0",
        "--mjlab_terminate_on_success=True", "--mjlab_success_bonus=50.0",
        "--mjlab_success_max_object_speed=0.15",
        f"--mjlab_descriptor_mode={mode}",
        "--mjlab_grip_tilt_min_degrees=15.0",
        "--mjlab_grip_tilt_max_degrees=45.0",
        "--mjlab_elbow_extension_min_degrees=45.0",
        "--mjlab_elbow_extension_max_degrees=65.0",
        "--mjlab_transport_start_distance=0.03",
        "--mjlab_approach_deviation_reference=0.05",
        "--mjlab_transport_deviation_reference=0.15",
        "--mjlab_contact_height_reference=0.02",
        "--num_dims=2", f"--grid_size={args.grid_size}",
        f"--seed={args.seed}", "--rollout_length=24",
        f"--env_batch_size={args.env_batch_size}",
        f"--popsize={args.popsize}", "--anneal_lr=False",
        "--num_minibatches=4", "--update_epochs=5",
        "--norm_adv_per_minibatch=False", "--mixed_precision=False",
        "--learning_rate=0.0001", "--vf_coef=1.0",
        "--entropy_coef=0.005", "--target_kl=0.01",
        "--adaptive_kl=True", "--max_grad_norm=1.0",
        "--action_transform=none", "--action_std_parameterization=direct",
        "--initial_action_std=0.5", "--actor_hidden_dims", "512", "256", "128",
        "--actor_activation=elu", "--value_bootstrap=True",
        "--eval_deterministic=True", "--eval_common_random_numbers=True",
        "--normalize_obs=True", "--normalize_returns=False",
        f"--total_iterations={args.iterations}", "--dqd_algorithm=cma_maega",
        f"--sigma0={args.sigma0}", "--xnes_center_init=zero",
        "--restart_rule=no_improvement", "--calc_gradient_iters=10",
        "--move_mean_iters=1", "--archive_lr=0.1", "--threshold_min=0",
        f"--archive_min_success_rate={args.min_success_rate}",
        f"--mean_min_success_rate={args.min_success_rate}",
        "--log_arch_freq=5", "--save_scheduler=True",
        "--save_heatmaps=True", "--heatmap_freq=10", "--use_wandb=False",
        "--wandb_project=ppga", "--wandb_group=mjlab_lift_cube_descriptors",
        f"--wandb_run_name={mode}_seed_{args.seed}",
        f"--expdir={args.output_dir / mode}",
    ]


def validation_command(args, checkpoint: Path) -> list[str]:
    return [
        sys.executable, "-u", "-m", "ppga.RL.train_ppo",
        "--env_type=mjlab", "--env_name=lift_cube", "--num_dims=2",
        f"--seed={args.seed}", f"--env_batch_size={args.env_batch_size}",
        "--rollout_length=24", "--total_timesteps=0", "--num_minibatches=4",
        "--update_epochs=5", "--learning_rate=0.0001", "--anneal_lr=False",
        "--target_kl=0.01", "--adaptive_kl=True", "--normalize_obs=True",
        "--normalize_returns=False", "--mixed_precision=False",
        "--action_transform=none", "--action_std_parameterization=direct",
        "--initial_action_std=0.5", "--actor_hidden_dims", "512", "256", "128",
        "--actor_activation=elu", "--value_bootstrap=True",
        "--eval_deterministic=True", "--eval_common_random_numbers=True",
        "--mjlab_fixed_goal=False", "--mjlab_disable_curriculum=True",
        "--mjlab_command_resampling_time=40.0",
        "--mjlab_terminate_on_success=True", "--mjlab_success_bonus=50.0",
        "--mjlab_success_max_object_speed=0.15",
        "--mjlab_descriptor_mode=contact_azimuth_height",
        f"--initial_actor_checkpoint={checkpoint}", "--use_wandb=False",
        "--checkpoint_interval_updates=0",
        f"--expdir={args.output_dir / 'bootstrap_validation'}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="Resume the first descriptor mode from this scheduler pickle, "
             "then run the remaining modes normally.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--hours-per-experiment", type=float, default=2.25)
    parser.add_argument("--env-batch-size", type=int, default=960)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--sigma0", type=float, default=0.02)
    parser.add_argument("--min-success-rate", type=float, default=0.75)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
        if not args.resume_checkpoint.is_file():
            parser.error(
                f"resume checkpoint does not exist: {args.resume_checkpoint}")
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --resume-checkpoint")
    if (args.bootstrap_iterations < 1 or args.bootstrap_envs < 1
            or args.iterations < 1 or args.popsize < 1 or args.grid_size < 2
            or not math.isfinite(args.hours_per_experiment)
            or args.hours_per_experiment <= 0 or args.sigma0 <= 0
            or not 0 <= args.min_success_rate <= 1):
        parser.error("invalid positive iteration, environment, time, or threshold value")
    if (args.env_batch_size < 1
            or args.env_batch_size % args.popsize
            or args.env_batch_size % 3):
        parser.error("env-batch-size must be divisible by popsize and 3")
    existing = ([] if not args.output_dir.exists() else [
        path for path in args.output_dir.iterdir()
        if path.name != "sequence.log"
    ])
    if args.resume_checkpoint is None and existing:
        parser.error(f"output directory is not empty: {args.output_dir}")
    if args.resume_checkpoint is not None:
        if not args.output_dir.is_dir():
            parser.error(f"resume output directory does not exist: {args.output_dir}")
        conflicts = [
            args.output_dir / "resume_sequence_state.json",
            args.output_dir / "ppga_contact_azimuth_height_resume.log",
            args.output_dir / "approach_orientation",
            args.output_dir / "ppga_approach_orientation.log",
            args.output_dir / "contact_transport",
            args.output_dir / "ppga_contact_transport.log",
        ]
        conflicts = [path for path in conflicts if path.exists()]
        if conflicts:
            parser.error(
                "resume outputs already exist: "
                + ", ".join(str(path) for path in conflicts))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[item for item in sys.argv[1:] if item != "--detach"]]
        with (args.output_dir / "sequence.log").open("x") as log:
            child = subprocess.Popen(
                command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Supervisor PID: {child.pid}\nOutput: {args.output_dir}", flush=True)
        return

    state = {
        "supervisor_pid": os.getpid(), "status": "starting",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed, "descriptor_modes": list(DESCRIPTOR_MODES),
        "env_batch_size": args.env_batch_size, "popsize": args.popsize,
        "iterations": args.iterations, "stages": [],
    }
    if args.resume_checkpoint is not None:
        state["resumed_from"] = str(args.resume_checkpoint)
    stop_requested: list[int] = []

    def request_stop(signum, _frame):
        stop_requested.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    def save(**updates) -> None:
        state.update(updates)
        target = args.output_dir / (
            "resume_sequence_state.json"
            if args.resume_checkpoint is not None else "sequence_state.json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(target)

    environment = dict(
        os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        MPLCONFIGDIR=str(args.output_dir / ".matplotlib"))
    (args.output_dir / ".matplotlib").mkdir(exist_ok=True)

    def run_stage(name: str, command: list[str], timeout_hours=None) -> tuple[int, bool]:
        log_path = args.output_dir / f"{name}.log"
        with log_path.open("x") as log:
            child = subprocess.Popen(
                command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            save(status=name, child_pid=child.pid, command=command)
            started = time.monotonic()
            stop_sent = None
            timed_out = False
            while child.poll() is None:
                if stop_requested or (timeout_hours is not None and
                                      time.monotonic() - started >= timeout_hours * 3600):
                    timed_out = not stop_requested
                    if stop_sent is None:
                        child.send_signal(signal.SIGTERM)
                        stop_sent = time.monotonic()
                        save(status="stopping", stop_reason=(
                            "user_request" if stop_requested else "stage_timeout"))
                    elif time.monotonic() - stop_sent > 900:
                        child.kill()
                time.sleep(5)
        state["stages"].append({
            "name": name, "exit_code": child.returncode,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
        })
        save(child_pid=None)
        return child.returncode, timed_out

    save()
    checkpoint = args.checkpoint
    if checkpoint is None:
        bootstrap_root = args.output_dir / "bootstrap_rsl"
        command = [
            sys.executable, "-u", "-m", "mjlab.scripts.train",
            "Mjlab-Lift-Cube-Yam",
            f"--env.scene.num-envs={args.bootstrap_envs}",
            f"--env.seed={args.seed}", f"--agent.seed={args.seed}",
            f"--agent.max-iterations={args.bootstrap_iterations}",
            "--agent.logger=tensorboard", "--agent.upload-model=False",
            f"--agent.run-name=descriptor_bootstrap_seed_{args.seed}",
            f"--log-root={bootstrap_root}",
        ]
        exit_code, _ = run_stage("bootstrap_training", command)
        if stop_requested or exit_code:
            save(status="stopped_by_request" if stop_requested else "failed",
                 failed_stage="bootstrap_training")
            return
        candidates = sorted(
            bootstrap_root.rglob("model_*.pt"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            save(status="failed", failed_stage="bootstrap_checkpoint_missing")
            return
        native_checkpoint = candidates[-1]
        checkpoint = args.output_dir / "bootstrap_actor.pt"
        exit_code, _ = run_stage(
            "bootstrap_conversion",
            [sys.executable, "-u", "-m", "ppga.RL.convert_mjlab_rsl_checkpoint",
             str(native_checkpoint), str(checkpoint)])
        if exit_code:
            save(status="failed", failed_stage="bootstrap_conversion")
            return

    assert checkpoint is not None
    if args.resume_checkpoint is None:
        exit_code, _ = run_stage(
            "bootstrap_validation", validation_command(args, checkpoint))
        if stop_requested or exit_code:
            save(status="stopped_by_request" if stop_requested else "failed",
                 failed_stage="bootstrap_validation")
            return
        evaluation_path = (args.output_dir / "bootstrap_validation"
                           / str(args.seed) / "evaluation.json")
        evaluation = json.loads(evaluation_path.read_text())
        success_rate = float(evaluation["metadata"]["episode_success_rate"])
        save(bootstrap_checkpoint=str(checkpoint), bootstrap_success_rate=success_rate)
        if success_rate < args.min_success_rate:
            save(status="bootstrap_rejected", failed_stage="bootstrap_validation")
            return
    else:
        save(bootstrap_checkpoint=str(checkpoint), bootstrap_validation="reused")

    for mode_index, mode in enumerate(DESCRIPTOR_MODES):
        if stop_requested:
            save(status="stopped_by_request")
            return
        scheduler_checkpoint = (
            args.resume_checkpoint if mode_index == 0 else None)
        stage_name = f"ppga_{mode}"
        if scheduler_checkpoint is not None:
            stage_name += "_resume"
        exit_code, timed_out = run_stage(
            stage_name,
            ppga_command(args, mode, checkpoint, scheduler_checkpoint),
            timeout_hours=args.hours_per_experiment)
        if stop_requested:
            save(status="stopped_by_request")
            return
        if exit_code and not timed_out:
            save(status="failed", failed_stage=f"ppga_{mode}")
            return

    save(status="complete", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


if __name__ == "__main__":
    main()
