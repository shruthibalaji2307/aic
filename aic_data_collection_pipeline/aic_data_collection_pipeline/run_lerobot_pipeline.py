import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _default_lerobot_dataset_root(dataset_repo_id: str) -> Path:
    """Match lerobot: HF_LEROBOT_HOME / repo_id (see lerobot.utils.constants)."""
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    lerobot_home = Path(os.environ.get("LEROBOT_HOME", str(hf_home / "lerobot")))
    return lerobot_home / dataset_repo_id


def _check_lerobot_dataset_dir(dataset_repo_id: str, resume: bool) -> str | None:
    """Return an error message, or None if the on-disk state is OK for this mode."""
    root = _default_lerobot_dataset_root(dataset_repo_id)
    info = root / "meta" / "info.json"

    if resume:
        if not root.exists():
            return (
                f"--lerobot-resume requires an existing dataset under {root}. "
                "Run once without --lerobot-resume first, or drop --lerobot-resume."
            )
        if not info.is_file():
            return (
                f"Cannot resume: {info} is missing (folder left from a failed run).\n"
                f"Remove the broken directory and record fresh without --resume:\n"
                f"  rm -rf {root}"
            )
        return None

    if root.exists():
        if info.is_file():
            return (
                f"A dataset already exists at {root}.\n"
                "Add episodes with --lerobot-resume, or remove and start over:\n"
                f"  rm -rf {root}"
            )
        return (
            f"Incomplete dataset directory (missing meta/info.json): {root}\n"
            f"Remove it before recording:\n"
            f"  rm -rf {root}"
        )
    return None


def _run_cmd(
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.Popen:
    print(f"\n[launch] {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, cwd=str(cwd) if cwd else None)


def _wait_or_terminate(
    proc: subprocess.Popen,
    name: str,
    timeout_sec: float,
) -> None:
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return
    if proc.returncode not in (None, 0):
        raise RuntimeError(f"{name} exited early with code {proc.returncode}")


def _terminate(
    proc: subprocess.Popen | None,
    name: str,
    sigint_wait_sec: float = 12.0,
) -> None:
    if not proc or proc.poll() is not None:
        return
    print(f"[stop] {name}")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=sigint_wait_sec)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def _pixi_ros2_pkg_available(workspace: Path, package: str) -> bool:
    r = subprocess.run(
        ["pixi", "run", "ros2", "pkg", "prefix", package],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _bash_ros_exec(setup_bash: Path, ros_argv: list[str]) -> list[str]:
    """Run ros2 via bash after sourcing a colcon/overlay install/setup.bash."""
    quoted_setup = shlex.quote(str(setup_bash.resolve()))
    quoted_cmd = " ".join(shlex.quote(a) for a in ros_argv)
    return ["bash", "-lc", f"set -e; source {quoted_setup} && exec {quoted_cmd}"]


def _ros2_cmd(
    workspace: Path,
    ros_argv: list[str],
    ros_setup_bash: Path | None,
) -> list[str]:
    if ros_setup_bash is not None:
        return _bash_ros_exec(ros_setup_bash, ros_argv)
    return ["pixi", "run", "ros2", *ros_argv]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single data-collection session using a generated AIC engine config, "
            "CheatCode policy, and lerobot-record."
        )
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path.cwd(),
        help="Root Pixi workspace directory (defaults to current directory).",
    )
    parser.add_argument(
        "--engine-config",
        type=Path,
        required=True,
        help="Path to generated AIC engine YAML config.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help=(
            "LeRobot dataset.repo_id (e.g. my-org/aic_demo_data on Hugging Face, or "
            "local/aic_session for disk-only). Required for full host-colcon mode; "
            "with --eval-in-container defaults to local/aic_cable_insert if omitted."
        ),
    )
    parser.add_argument(
        "--dataset-single-task",
        default="insert cable plug into target port",
        help="LeRobot dataset single-task prompt.",
    )
    parser.add_argument(
        "--dataset-push-to-hub",
        action="store_true",
        help="Set --dataset.push_to_hub=true for lerobot-record.",
    )
    parser.add_argument(
        "--dataset-private",
        action="store_true",
        help="Set --dataset.private=true for lerobot-record.",
    )
    parser.add_argument(
        "--teleop-type",
        choices=["aic_keyboard_ee", "aic_spacemouse", "aic_keyboard_joint"],
        default="aic_keyboard_ee",
        help="Teleoperator type passed to lerobot-record.",
    )
    parser.add_argument(
        "--robot-teleop-target-mode",
        choices=["cartesian", "joint"],
        default="cartesian",
        help="Robot target mode passed to lerobot-record.",
    )
    parser.add_argument(
        "--robot-teleop-frame-id",
        choices=["base_link", "gripper/tcp"],
        default="base_link",
        help="Robot frame id passed to lerobot-record.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.home() / "aic_results" / "lerobot_pipeline",
        help="Directory for AIC engine result outputs.",
    )
    parser.add_argument(
        "--startup-wait-sec",
        type=float,
        default=8.0,
        help="Warmup wait before starting lerobot-record.",
    )
    parser.add_argument(
        "--ros-setup-bash",
        type=Path,
        default=None,
        help=(
            "Path to a ROS 2 overlay install/setup.bash (e.g. colcon install) that "
            "contains aic_bringup, aic_engine, Gazebo bringup, etc. Required unless "
            "those packages are installed in the Pixi environment."
        ),
    )
    parser.add_argument(
        "--start-zenoh-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start rmw_zenoh_cpp rmw_zenohd before other ROS nodes (default: true).",
    )
    parser.add_argument(
        "--eval-in-container",
        action="store_true",
        help=(
            "Use when Gazebo + aic_engine already run inside the recommended "
            "`aic_eval` image (e.g. Distrobox /entrypoint.sh). Skips launching "
            "Zenoh, aic_model, and aic_bringup from this script; only runs "
            "`pixi run lerobot-record` on the host."
        ),
    )
    parser.add_argument(
        "--launch-cheatcode-on-host",
        action="store_true",
        help=(
            "With --eval-in-container, also start CheatCode via "
            "`pixi run ros2 run aic_model ...` on the host before recording "
            "(engine waits for aic_model; start it within the engine timeout)."
        ),
    )
    parser.add_argument(
        "--lerobot-resume",
        action="store_true",
        help=(
            "Pass --resume=true to lerobot-record: append to an existing dataset at "
            "~/.cache/huggingface/lerobot/<repo_id> instead of creating a new folder "
            "(avoids FileExistsError when reusing the same --dataset-repo-id)."
        ),
    )
    return parser.parse_args()


def _lerobot_record_cmd(
    workspace: Path,
    dataset_repo_id: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "pixi",
        "run",
        "lerobot-record",
        "--robot.type=aic_controller",
        "--robot.id=aic",
        f"--teleop.type={args.teleop_type}",
        "--teleop.id=aic",
        f"--robot.teleop_target_mode={args.robot_teleop_target_mode}",
        f"--robot.teleop_frame_id={args.robot_teleop_frame_id}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.single_task={args.dataset_single_task}",
        f"--dataset.push_to_hub={'true' if args.dataset_push_to_hub else 'false'}",
        f"--dataset.private={'true' if args.dataset_private else 'false'}",
        "--play_sounds=false",
        "--display_data=true",
    ]
    if args.lerobot_resume:
        cmd.append("--resume=true")
    return cmd


def _resolve_dataset_repo_id(args: argparse.Namespace) -> str | None:
    if args.dataset_repo_id:
        return args.dataset_repo_id
    if args.eval_in_container:
        return "local/aic_cable_insert"
    return None


def _run_lerobot_record_only(
    workspace: Path,
    env: dict[str, str],
    args: argparse.Namespace,
    dataset_repo_id: str,
) -> int:
    """Host-side recording while eval stack runs in aic_eval container."""
    model_proc: subprocess.Popen | None = None
    if args.launch_cheatcode_on_host:
        model_cmd = [
            "pixi",
            "run",
            "ros2",
            "run",
            "aic_model",
            "aic_model",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "policy:=aic_example_policies.ros.CheatCode",
        ]
        model_proc = _run_cmd(model_cmd, env=env, cwd=workspace)
        _wait_or_terminate(model_proc, "aic_model", 3.0)

    try:
        time.sleep(args.startup_wait_sec)
        record_cmd = _lerobot_record_cmd(workspace, dataset_repo_id, args)
        print(
            "\n[info] lerobot-record key controls: Right Arrow=next episode, "
            "Left Arrow=redo episode, ESC=stop."
        )
        record_proc = _run_cmd(record_cmd, env=env, cwd=workspace)
        try:
            record_code = record_proc.wait()
            if record_code != 0:
                print(
                    f"[error] lerobot-record exited with code {record_code}. "
                    "If you saw FileExistsError on the dataset folder, remove it or "
                    "re-run with --lerobot-resume.",
                    file=sys.stderr,
                )
                return 1
        finally:
            _terminate(record_proc, "lerobot-record")
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        return 130
    finally:
        _terminate(model_proc, "aic_model", sigint_wait_sec=15.0)

    return 0


def main() -> int | None:
    args = _parse_args()
    workspace = args.workspace_dir.resolve()
    config_path = args.engine_config.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_repo_id = _resolve_dataset_repo_id(args)
    if dataset_repo_id is None:
        print(
            "error: --dataset-repo-id is required unless you use --eval-in-container "
            "(which defaults to local/aic_cable_insert).\n"
            "Example: --dataset-repo-id your_hf_user/aic-cable-insert-smoke\n"
            "Or local-only: --dataset-repo-id local/my_dataset_name",
            file=sys.stderr,
        )
        return 1

    dir_err = _check_lerobot_dataset_dir(dataset_repo_id, args.lerobot_resume)
    if dir_err:
        print(f"error: {dir_err}", file=sys.stderr)
        return 1

    if args.launch_cheatcode_on_host and not args.eval_in_container:
        print(
            "error: --launch-cheatcode-on-host requires --eval-in-container.",
            file=sys.stderr,
        )
        return 1

    if args.eval_in_container:
        env = os.environ.copy()
        env["AIC_RESULTS_DIR"] = str(results_dir)
        print(
            "\n[info] --eval-in-container: ensure `aic_eval` is already running with a matching "
            "aic_engine_config_file (see README). Config path for your reference:",
            config_path,
        )
        if not args.dataset_repo_id:
            print(
                f"[info] Using default dataset.repo_id={dataset_repo_id} "
                "(pass --dataset-repo-id to override).",
            )
        try:
            return _run_lerobot_record_only(workspace, env, args, dataset_repo_id) or 0
        finally:
            print("\nDone.")
            print(f"- Engine artifacts directory (host): {results_dir}")
            print(f"- Engine config (use this path in container entrypoint): {config_path}")

    ros_setup = args.ros_setup_bash.resolve() if args.ros_setup_bash else None
    if ros_setup is not None and not ros_setup.is_file():
        print(
            f"error: --ros-setup-bash must point to an existing file: {ros_setup}",
            file=sys.stderr,
        )
        return 1

    if ros_setup is None and not _pixi_ros2_pkg_available(workspace, "aic_bringup"):
        print(
            "error: package 'aic_bringup' is not available inside `pixi run ros2`.\n"
            "The default Pixi workspace does not include the Gazebo/simulation stack.\n"
            "Fix one of:\n"
            "  A) Recommended: run sim in Distrobox + aic_eval (see package README), then on the host run:\n"
            "       aic-run-lerobot-pipeline --eval-in-container ...\n"
            "  B) Advanced: colcon build locally and pass:\n"
            "       --ros-setup-bash /path/to/install/setup.bash\n"
            "Also start a Zenoh router if you are not already (this script does by default when not using the container).",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    env["AIC_RESULTS_DIR"] = str(results_dir)

    zenoh_proc: subprocess.Popen | None = None
    if args.start_zenoh_router:
        zenoh_cmd = _ros2_cmd(
            workspace,
            ["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
            ros_setup,
        )
        zenoh_proc = _run_cmd(zenoh_cmd, env=env, cwd=workspace)
        time.sleep(1.5)

    # 1) Launch policy node with CheatCode.
    model_cmd = _ros2_cmd(
        workspace,
        [
            "ros2",
            "run",
            "aic_model",
            "aic_model",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "policy:=aic_example_policies.ros.CheatCode",
        ],
        ros_setup,
    )
    model_proc = _run_cmd(model_cmd, env=env, cwd=workspace)
    try:
        _wait_or_terminate(model_proc, "aic_model", 3.0)

        # 2) Launch simulation + engine with generated config.
        sim_cmd = _ros2_cmd(
            workspace,
            [
                "ros2",
                "launch",
                "aic_bringup",
                "aic_gz_bringup.launch.py",
                "ground_truth:=true",
                "start_aic_engine:=true",
                "shutdown_on_aic_engine_exit:=true",
                f"aic_engine_config_file:={config_path}",
            ],
            ros_setup,
        )
        sim_proc = _run_cmd(sim_cmd, env=env, cwd=workspace)
        try:
            _wait_or_terminate(sim_proc, "aic_gz_bringup", 8.0)
            time.sleep(args.startup_wait_sec)

            # 3) Start lerobot-record with the command pattern from lerobot_robot_aic/README.md.
            record_cmd = _lerobot_record_cmd(workspace, dataset_repo_id, args)
            print(
                "\n[info] lerobot-record key controls: Right Arrow=next episode, "
                "Left Arrow=redo episode, ESC=stop."
            )
            record_proc = _run_cmd(record_cmd, env=env, cwd=workspace)
            try:
                record_code = record_proc.wait()
                if record_code != 0:
                    print(
                        f"[error] lerobot-record exited with code {record_code}. "
                        "If the dataset folder already exists, re-run with --lerobot-resume "
                        "or delete ~/.cache/huggingface/lerobot/<your_repo_id>.",
                        file=sys.stderr,
                    )
                    return 1
            finally:
                _terminate(record_proc, "lerobot-record")
        finally:
            _terminate(sim_proc, "aic_gz_bringup")
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        return 130
    finally:
        _terminate(model_proc, "aic_model", sigint_wait_sec=15.0)
        _terminate(zenoh_proc, "rmw_zenohd")

    print("\nDone.")
    print(f"- Engine artifacts directory: {results_dir}")
    print(f"- Engine config used: {config_path}")
    print(
        "- If dataset.push_to_hub=false, push later with huggingface-cli or "
        "rerun with --dataset-push-to-hub."
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
