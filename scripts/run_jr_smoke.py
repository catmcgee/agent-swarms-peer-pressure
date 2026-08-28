from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from swarmstop.boards import BoardBank
from swarmstop.config import (
    load_experiment_config,
    load_peer_boards,
    load_tasks,
    validate_experiment,
)
from swarmstop.io import ResultWriter
from swarmstop.lens_runtime import extract_lens_records
from swarmstop.mechanistic import (
    SnapshotWriter,
    load_concept_registry,
    load_mechanistic_config,
)
from swarmstop.model import TransformersToolModel
from swarmstop.runner import ControlledTrialRunner
from swarmstop.schema import stable_trial_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-config", default="configs/jr_smoke.yaml")
    parser.add_argument("--lens-config", default="configs/jr_lens.yaml")
    parser.add_argument("--output-dir", default="results/jr-lens-runpod-smoke")
    parser.add_argument("--skip-behavior", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = output_dir / "anchor-snapshots.jsonl"

    lens_config = load_mechanistic_config(args.lens_config)
    registry = load_concept_registry(lens_config.registry_path)
    behavior_config = replace(
        load_experiment_config(args.behavior_config),
        output_dir=str(output_dir / "behavior"),
    )
    tasks = load_tasks(behavior_config.tasks_path)
    boards = load_peer_boards(behavior_config.peer_boards_path)
    validate_experiment(behavior_config, tasks, boards)

    model = TransformersToolModel(
        lens_config.model_id,
        revision=lens_config.model_revision or "",
        tokenizer_revision=lens_config.tokenizer_revision,
    )
    if not args.skip_behavior:
        _run_behavior(
            config=behavior_config,
            tasks=tasks,
            boards=boards,
            model=model,
            snapshots_path=snapshots_path,
            revision=lens_config.model_revision or "",
        )
    if not snapshots_path.exists():
        raise ValueError(f"snapshot file does not exist: {snapshots_path}")

    summary = extract_lens_records(
        config=lens_config,
        snapshots_path=snapshots_path,
        output_dir=output_dir / "lens",
        model=model.model,
        tokenizer=model.tokenizer,
        registry=registry,
    )
    print(json.dumps({"status": "complete", **summary}, sort_keys=True))


def _run_behavior(*, config, tasks, boards, model, snapshots_path: Path, revision: str) -> None:
    writer = ResultWriter(config.output_dir, config)
    snapshot_writer = SnapshotWriter(snapshots_path)
    bank = BoardBank(boards)
    runner = ControlledTrialRunner(
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        snapshot_observer=snapshot_writer,
        model_revision=revision,
    )
    attempted = 0
    for task in tasks:
        for condition in config.conditions:
            for seed in config.seeds:
                board = bank.choose(task_id=task.id, norm=condition.peer_norm, seed=seed)
                trial_id = stable_trial_id(
                    task_id=task.id,
                    condition=condition,
                    model=model.model_id,
                    seed=seed,
                    board_id=board.id if board else None,
                )
                if trial_id in writer.completed:
                    if not snapshot_writer.has_complete_trial(trial_id):
                        raise ValueError(f"completed trial is missing snapshots: {trial_id}")
                    continue
                result = runner.run(
                    task=task,
                    condition=condition,
                    model=model,
                    seed=seed,
                    board=board,
                )
                writer.append(result)
                attempted += 1
                print(
                    json.dumps(
                        {"phase": "behavior", "completed": attempted, "trial_id": trial_id}
                    )
                )


if __name__ == "__main__":
    main()
