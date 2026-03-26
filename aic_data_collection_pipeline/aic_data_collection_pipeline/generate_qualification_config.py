import argparse
import copy
import random
from pathlib import Path

import yaml


def _randomize_board_pose(trial: dict, rng: random.Random) -> None:
    trial["scene"]["task_board"]["pose"]["x"] = round(rng.uniform(0.12, 0.20), 4)
    trial["scene"]["task_board"]["pose"]["y"] = round(rng.uniform(-0.22, 0.05), 4)
    trial["scene"]["task_board"]["pose"]["yaw"] = round(rng.uniform(2.85, 3.20), 4)


def _default_nic_entity_pose() -> dict:
    """Default NIC rail pose when template only has entity_present: false (no entity_pose)."""
    return {"translation": 0.036, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}


def _default_sc_entity_pose() -> dict:
    """Default SC rail pose when the inactive rail has no entity_pose in the template."""
    return {"translation": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}


def _build_sfp_trial(template_trial: dict, rng: random.Random) -> dict:
    trial = copy.deepcopy(template_trial)
    task_board = trial["scene"]["task_board"]
    task = trial["tasks"]["task_1"]

    target_nic_rail = rng.choice([0, 1, 2, 3, 4])
    target_port = rng.choice(["sfp_port_0", "sfp_port_1"])

    for rail_idx in range(5):
        rail_key = f"nic_rail_{rail_idx}"
        block = task_board[rail_key]
        block["entity_present"] = rail_idx == target_nic_rail
        if rail_idx == target_nic_rail:
            block["entity_name"] = f"nic_card_{rail_idx}"
            if "entity_pose" not in block:
                block["entity_pose"] = copy.deepcopy(_default_nic_entity_pose())
            block["entity_pose"]["translation"] = round(
                rng.uniform(-0.021, 0.023), 4
            )
            block["entity_pose"]["yaw"] = round(rng.uniform(-0.18, 0.18), 4)

    task["port_name"] = target_port
    task["target_module_name"] = f"nic_card_mount_{target_nic_rail}"
    _randomize_board_pose(trial, rng)
    return trial


def _build_sc_trial(template_trial: dict, rng: random.Random) -> dict:
    trial = copy.deepcopy(template_trial)
    task_board = trial["scene"]["task_board"]
    task = trial["tasks"]["task_1"]

    target_sc_rail = rng.choice([0, 1])
    for rail_idx in [0, 1]:
        rail_key = f"sc_rail_{rail_idx}"
        block = task_board[rail_key]
        block["entity_present"] = rail_idx == target_sc_rail
        if rail_idx == target_sc_rail:
            block["entity_name"] = f"sc_mount_{rail_idx}"
            if "entity_pose" not in block:
                block["entity_pose"] = copy.deepcopy(_default_sc_entity_pose())
            block["entity_pose"]["translation"] = round(
                rng.uniform(-0.055, 0.055), 4
            )

    task["target_module_name"] = f"sc_port_{target_sc_rail}"
    _randomize_board_pose(trial, rng)
    return trial


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an AIC engine qualification-style config file with randomized "
            "SFP and SC insertion tasks."
        )
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        required=True,
        help="Path to base template config (e.g. aic_engine/config/sample_config.yaml).",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
        help="Path to write generated config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of trials to generate.",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "alternating"],
        default="random",
        help=(
            "random: sample SFP/SC each trial; alternating: interleave SFP and SC for "
            "balanced output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    config = yaml.safe_load(args.template_config.read_text())
    template_sfp = config["trials"]["trial_1"]
    template_sc = config["trials"]["trial_3"]

    generated_trials = {}
    for idx in range(args.num_trials):
        trial_name = f"trial_{idx + 1}"
        if args.mode == "alternating":
            task_type = "sfp" if idx % 2 == 0 else "sc"
        else:
            task_type = rng.choice(["sfp", "sc"])

        if task_type == "sfp":
            generated_trials[trial_name] = _build_sfp_trial(template_sfp, rng)
        else:
            generated_trials[trial_name] = _build_sc_trial(template_sc, rng)

    config["trials"] = generated_trials

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Generated config: {args.output_config}")
    print(f"Trials: {len(generated_trials)} | Seed: {args.seed} | Mode: {args.mode}")


if __name__ == "__main__":
    main()
