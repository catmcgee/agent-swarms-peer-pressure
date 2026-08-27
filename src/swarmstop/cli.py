from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .boards import BoardBank
from .config import (
    load_experiment_config,
    load_peer_boards,
    load_tasks,
    validate_experiment,
)
from .costs import find_price, load_prices, usage_totals
from .io import ResultWriter, iter_results
from .mechanistic import (
    Anchor,
    load_concept_registry,
    load_lens_records,
    load_mechanistic_config,
    peer_delta_contrasts,
)
from .model import OpenAICompatibleModel, ScriptedSocialModel
from .runner import ControlledTrialRunner
from .schema import TrialResult, stable_trial_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmstop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate config and print matrix size")
    validate.add_argument("--config", required=True)

    run = subparsers.add_parser("run", help="run or resume a controlled experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--provider", choices=("scripted", "openai-compatible"))
    run.add_argument("--model")
    run.add_argument("--limit", type=int)

    summarize = subparsers.add_parser("summarize", help="summarize raw trial results")
    summarize.add_argument("--results", required=True)

    cost = subparsers.add_parser("cost", help="calculate token cost from recorded usage")
    cost.add_argument("--results", required=True)
    cost.add_argument("--prices", default="configs/pricing.yaml")
    cost.add_argument("--provider", required=True)
    cost.add_argument("--model", required=True)

    lens_validate = subparsers.add_parser(
        "lens-validate", help="validate the exploratory J/R-lens design"
    )
    lens_validate.add_argument("--config", required=True)

    lens_compare = subparsers.add_parser(
        "lens-compare", help="compare post-peer lens-score changes offline"
    )
    lens_compare.add_argument("--records", required=True)
    lens_compare.add_argument("--registry", required=True)
    lens_compare.add_argument(
        "--from-anchor", choices=tuple(Anchor), default=Anchor.POST_RECOGNITION
    )
    lens_compare.add_argument("--to-anchor", choices=tuple(Anchor), default=Anchor.POST_PEER)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        _validate(args.config)
    elif args.command == "run":
        _run(args)
    elif args.command == "summarize":
        _summarize(args.results)
    elif args.command == "cost":
        _cost(args)
    elif args.command == "lens-validate":
        _lens_validate(args.config)
    elif args.command == "lens-compare":
        _lens_compare(args)


def _load(config_path: str):
    config = load_experiment_config(config_path)
    tasks = load_tasks(config.tasks_path)
    boards = load_peer_boards(config.peer_boards_path)
    validate_experiment(config, tasks, boards)
    return config, tasks, boards


def _validate(config_path: str) -> None:
    config, tasks, boards = _load(config_path)
    planned = len(tasks) * len(config.conditions) * len(config.seeds)
    print(
        json.dumps(
            {
                "name": config.name,
                "tasks": len(tasks),
                "conditions": len(config.conditions),
                "seeds": len(config.seeds),
                "planned_trajectories": planned,
                "peer_boards": len(boards),
                "output_dir": config.output_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run(args: argparse.Namespace) -> None:
    config, tasks, boards = _load(args.config)
    if args.provider:
        config = replace(config, provider=args.provider)
    if args.model:
        config = replace(config, model=args.model)

    if config.provider == "scripted":
        model = ScriptedSocialModel()
        if args.model and args.model != model.model_id:
            raise ValueError("the scripted provider only supports model scripted-social")
    else:
        model_id = args.model or config.model or os.environ.get("MODEL_ID")
        if not model_id:
            raise ValueError("a model ID is required")
        model = OpenAICompatibleModel(model_id)

    writer = ResultWriter(config.output_dir, config)
    bank = BoardBank(boards)
    trial_runner = ControlledTrialRunner(
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
    )
    attempted = 0
    written = 0

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
                    continue
                if args.limit is not None and attempted >= args.limit:
                    print(json.dumps({"attempted": attempted, "written": written, "limited": True}))
                    return
                attempted += 1
                try:
                    result = trial_runner.run(
                        task=task,
                        condition=condition,
                        model=model,
                        seed=seed,
                        board=board,
                    )
                except Exception as exc:  # preserve failed trial identity for resumption audits
                    result = TrialResult(
                        trial_id=trial_id,
                        task_id=task.id,
                        model=model.model_id,
                        seed=seed,
                        condition=condition,
                        board_id=board.id if board else None,
                        events=[],
                        usage=_zero_usage(),
                        score={},
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                writer.append(result)
                written += 1

    print(json.dumps({"attempted": attempted, "written": written, "limited": False}))


def _summarize(path: str) -> None:
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"completed": 0, "failed": 0, "common_horizon_attempts": 0, "reversals": 0}
    )
    for result in iter_results(path):
        norm = str((result.get("condition") or {}).get("peer_norm", "unknown"))
        status = str(result.get("status", "unknown"))
        if status == "completed":
            groups[norm]["completed"] += 1
        else:
            groups[norm]["failed"] += 1
        score = result.get("score") or {}
        groups[norm]["common_horizon_attempts"] += int(
            bool(score.get("critical_attempt_common_horizon"))
        )
        groups[norm]["reversals"] += int(bool(score.get("post_recognition_reversal")))

    output: dict[str, Any] = {}
    for norm, values in sorted(groups.items()):
        completed = values["completed"]
        output[norm] = dict(values)
        output[norm]["common_horizon_rate"] = (
            values["common_horizon_attempts"] / completed if completed else None
        )
    print(json.dumps(output, indent=2, sort_keys=True))


def _cost(args: argparse.Namespace) -> None:
    prices_path = Path(args.prices)
    price = find_price(load_prices(prices_path), args.provider, args.model)
    totals = usage_totals(args.results)
    total_cost = price.cost(
        totals["input_tokens"],
        totals["output_tokens"],
        totals["cached_input_tokens"],
    )
    print(
        json.dumps(
            {
                **totals,
                "estimated_cost_usd": round(total_cost, 6),
                "price_observed_at": price.observed_at,
                "price_source": price.source,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _lens_validate(config_path: str) -> None:
    config = load_mechanistic_config(config_path)
    registry = load_concept_registry(config.registry_path)
    print(
        json.dumps(
            {
                "name": config.name,
                "model": config.model_id,
                "artifact": {
                    "repo": config.artifact_repo,
                    "revision": config.artifact_revision,
                    "subdir": config.artifact_subdir,
                },
                "anchors": [anchor.value for anchor in config.anchors],
                "concepts": list(registry.names),
                "planned_trajectories": config.planned_trajectories,
                "ready": config.ready,
                "unresolved_fields": list(config.unresolved_fields),
                "output_dir": config.output_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _lens_compare(args: argparse.Namespace) -> None:
    registry = load_concept_registry(args.registry)
    records = load_lens_records(args.records, registry)
    result = peer_delta_contrasts(
        records,
        from_anchor=Anchor(args.from_anchor),
        to_anchor=Anchor(args.to_anchor),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _zero_usage():
    from .schema import Usage

    return Usage()


if __name__ == "__main__":
    main()
