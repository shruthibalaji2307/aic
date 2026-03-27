#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


"""
CheatCode baseline benchmark — score tracking and report.

Runs the unmodified CheatCode policy for a fixed number of episodes and
prints a formatted score report at the end.  Use this to establish a
CheatCode baseline so that DataCollectorCheatCode improvements can be
quantified against it.

Usage (with the sim launched with ``ground_truth:=true``):

    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=aic_example_policies.ros.CheatCodeBenchmark \\
        -p num_episodes:=10

ROS parameters (set via ``--ros-args -p <name>:=<value>``):
    num_episodes  – number of episodes to run; -1 for unlimited
                    (default: -1). The report is only printed when
                    this limit is reached.
"""

import os
import re
import time
import threading

from rcl_interfaces.msg import Log

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .CheatCode import CheatCode


# Matches: ✓ Trial 'trial_3' completed successfully! Score: 39.375380
_TRIAL_COMPLETE_RE = re.compile(
    r"Trial '(trial_\d+)' completed successfully.*Score:\s*([\d.]+)"
)
# Strip ANSI escape codes
_ANSI_RE = re.compile(r"\033\[[0-9;]*[mK]")


class CheatCodeBenchmark(CheatCode):
    """Unmodified CheatCode with live score tally and a final breakdown report.

    Subscribes to ``/rosout`` to capture aic_engine per-trial score messages.
    - After each trial: prints a live one-line score update.
    - After all trials: prints a summary report with per-trial scores and
      aggregate statistics.

    Inherits CheatCode directly; ``insert_cable()`` is delegated to
    ``super()`` without any modification.
    """

    # ── ANSI colour helpers ───────────────────────────────────────────────
    _BOLD   = "\033[1m"
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _DIM    = "\033[2m"
    _RESET  = "\033[0m"

    def __init__(self, parent_node):
        super().__init__(parent_node)

        parent_node.declare_parameter("num_episodes", -1)

        self._num_episodes: int = (
            parent_node.get_parameter("num_episodes")
            .get_parameter_value()
            .integer_value
        )

        self._trial_index: int = 0          # completed trials this run
        self._run_id: str = time.strftime("%Y%m%d_%H%M%S")
        self._trial_tasks: list[str] = []   # task description per trial

        # Per-trial total scores received from /rosout
        self._live_trial_scores: dict[str, float] = {}

        self._rosout_sub = parent_node.create_subscription(
            Log, "/rosout", self._on_rosout, 100
        )

        self.get_logger().info(
            f"CheatCodeBenchmark init — run_id={self._run_id}, "
            f"num_episodes={self._num_episodes}"
        )

    # ── /rosout subscriber ────────────────────────────────────────────────

    def _on_rosout(self, msg: Log) -> None:
        """Parse aic_engine log messages for per-trial scores."""
        if msg.name != "aic_engine":
            return

        stripped = _ANSI_RE.sub("", msg.msg).strip()

        m = _TRIAL_COMPLETE_RE.search(stripped)
        if m:
            trial_id = m.group(1)
            score = float(m.group(2))
            self._live_trial_scores[trial_id] = score
            self._print_live_trial(trial_id, score)

    # ── Live tally ────────────────────────────────────────────────────────

    def _print_live_trial(self, trial_id: str, total_score: float) -> None:
        """Print a one-line update after each trial completes."""
        try:
            n = int(trial_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            n = 0

        progress = (
            f"{n}/{self._num_episodes}" if self._num_episodes > 0 else str(n)
        )
        task = self._trial_tasks[n - 1][:55] if 0 < n <= len(self._trial_tasks) else ""
        clr = (
            self._GREEN  if total_score > 80 else
            self._YELLOW if total_score > 40 else
            self._RED
        )
        print(
            f"\n{self._BOLD}{self._CYAN}  ▶ Trial {progress} complete{self._RESET}  "
            f"score={clr}{self._BOLD}{total_score:.2f}{self._RESET}  "
            f"{self._DIM}{task}{self._RESET}\n",
            flush=True,
        )

    # ── Task description helper ──────────────────────────────────────────

    @staticmethod
    def _task_to_string(task: Task) -> str:
        return (
            f"Insert {task.plug_type} plug ({task.cable_name}/{task.plug_name}) "
            f"into {task.port_type} port ({task.target_module_name}/{task.port_name})"
        )

    # ── Final report ─────────────────────────────────────────────────────

    def _print_benchmark_report(self) -> None:
        """Print a colour-coded benchmark report to the terminal."""
        B, G, R, Y, C, D, X = (
            self._BOLD, self._GREEN, self._RED,
            self._YELLOW, self._CYAN, self._DIM, self._RESET,
        )
        W = 60
        SEP = f"{D}{'─' * W}{X}"

        # Collect scores in trial order for this run
        scores: list[tuple[int, float]] = []
        for i in range(1, self._trial_index + 1):
            key = f"trial_{i}"
            if key in self._live_trial_scores:
                scores.append((i, self._live_trial_scores[key]))

        lines: list[str] = []
        lines.append("")
        lines.append(f"{B}{C}{'═' * W}{X}")
        lines.append(f"{B}{C}  CHEATCODE BENCHMARK REPORT{X}")
        lines.append(f"{B}{C}{'═' * W}{X}")

        # ── Run info ──────────────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  Run Info{X}")
        lines.append(SEP)
        lines.append(f"  Run ID:          {self._run_id}")
        lines.append(f"  Policy:          CheatCode (unmodified baseline)")
        lines.append(f"  Trials executed: {self._trial_index}")
        lines.append(f"  Trials scored:   {len(scores)}")

        # ── Per-trial scores ──────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  Per-Trial Scores{X}")
        lines.append(SEP)

        if scores:
            lines.append(f"  {D}{'Trial':>5}  {'Score':>8}  Task{X}")
            for i, score in scores:
                task_short = (
                    self._trial_tasks[i - 1] if i <= len(self._trial_tasks) else ""
                )[:40]
                clr = G if score > 80 else (Y if score > 40 else R)
                lines.append(
                    f"  {i:>5}  "
                    f"{clr}{score:>8.3f}{X}  "
                    f"{D}{task_short}{X}"
                )
        else:
            lines.append(f"  {Y}No scoring data received from /rosout.{X}")

        # ── Aggregate ─────────────────────────────────────────────────────
        if scores:
            vals = [s for _, s in scores]
            avg = sum(vals) / len(vals)
            clr_avg = G if avg > 80 else (Y if avg > 40 else R)
            successes = sum(1 for s in vals if s > 80)
            rate = successes / len(vals)
            clr_rate = G if rate >= 0.8 else (Y if rate >= 0.4 else R)

            lines.append(SEP)
            lines.append(f"{B}  Aggregate  ({len(vals)} trials){X}")
            lines.append(SEP)
            lines.append(
                f"  {'Average score:':.<28s} "
                f"{clr_avg}{B}{avg:.3f}{X}  "
                f"(range {min(vals):.2f}–{max(vals):.2f})"
            )
            lines.append(
                f"  {'High score rate (>80):':.<28s} "
                f"{clr_rate}{B}{successes}/{len(vals)} "
                f"({rate * 100:.0f}%){X}"
            )

        lines.append(f"{B}{C}{'═' * W}{X}")
        lines.append("")

        report = "\n".join(lines)
        print(report, flush=True)

        # Save a plain-text (ANSI-stripped) copy to the working directory.
        report_path = os.path.join(
            os.getcwd(), f"cheatcode_report_{self._run_id}.txt"
        )
        try:
            with open(report_path, "w") as f:
                f.write(_ANSI_RE.sub("", report))
            self.get_logger().info(f"Benchmark report saved to {report_path}")
        except OSError as exc:
            self.get_logger().warn(f"Could not save report to {report_path}: {exc}")

    # ── Policy entry point ────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        trial_label = (
            f"{self._trial_index + 1}"
            + (f"/{self._num_episodes}" if self._num_episodes > 0 else "")
        )
        self.get_logger().info(
            f"CheatCodeBenchmark.insert_cable() trial {trial_label}"
        )

        self._trial_tasks.append(self._task_to_string(task))

        # Run CheatCode exactly as-is — no wrapping, no modification.
        result = super().insert_cable(task, get_observation, move_robot, send_feedback)

        self._trial_index += 1

        is_final = (
            self._num_episodes > 0
            and self._trial_index >= self._num_episodes
        )
        if is_final:
            def _report_and_exit() -> None:
                # Per-trial scores arrive between trials (before lifecycle
                # teardown), but aic_engine takes ~2-3 s after the last
                # insert_cable() returns to finish scoring and log the score.
                # Wait long enough to capture the final trial's score.
                time.sleep(5.0)
                self._print_benchmark_report()
                self.get_logger().info("Benchmark report complete. Exiting.")
                time.sleep(1.0)
                os._exit(0)

            threading.Thread(target=_report_and_exit, daemon=True).start()
            threading.Timer(15.0, lambda: os._exit(0)).start()  # safety

            self.get_logger().info(
                "All benchmark episodes complete. Printing report..."
            )

        return result
