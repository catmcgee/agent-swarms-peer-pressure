import json
from pathlib import Path

from swarmstop.cli import main
from swarmstop.mechanistic import load_concept_registry

ROOT = Path(__file__).resolve().parents[1]


def test_cost_and_lens_validation_commands_produce_separate_reports(tmp_path: Path, capsys) -> None:
    results = tmp_path / "trials.jsonl"
    results.write_text(
        json.dumps(
            {
                "status": "completed",
                "usage": {
                    "input_tokens": 1_000_000,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(
        [
            "cost",
            "--results",
            str(results),
            "--prices",
            str(ROOT / "configs/pricing.yaml"),
            "--provider",
            "together",
            "--model",
            "Qwen/Qwen3.5-9B",
        ]
    )
    cost_report = json.loads(capsys.readouterr().out)

    main(["lens-validate", "--config", str(ROOT / "configs/jr_lens.yaml")])
    lens_report = json.loads(capsys.readouterr().out)

    assert cost_report["estimated_cost_usd"] == 0.1
    assert "planned_trajectories" not in cost_report
    assert lens_report["planned_trajectories"] == 216
    assert lens_report["ready"] is True
    assert "estimated_cost_usd" not in lens_report


def test_lens_compare_command_reports_peer_delta(tmp_path: Path, capsys) -> None:
    registry_path = ROOT / "configs/concepts/jr_lens.yaml"
    concepts = load_concept_registry(registry_path).names
    records_path = tmp_path / "lens-records.jsonl"
    records: list[dict] = []
    for trial_id, norm, before, after in (
        ("stop-1", "stop", 0.8, 0.9),
        ("proceed-1", "proceed", 0.8, 0.4),
    ):
        for anchor, boundary_score in (
            ("post_recognition", before),
            ("post_peer", after),
        ):
            scores = dict.fromkeys(concepts, 0.0)
            scores["boundary_denial"] = boundary_score
            records.append(
                {
                    "schema_version": 1,
                    "trial_id": trial_id,
                    "task_id": "task-1",
                    "task_family": "family-1",
                    "model_id": "model-1",
                    "model_revision": "revision-1",
                    "tokenizer_revision": "tokenizer-1",
                    "seed": 1,
                    "condition": {
                        "feasibility": "impossible",
                        "investment": "personal",
                        "peer_norm": norm,
                        "budget_rounds": 8,
                    },
                    "anchor": anchor,
                    "layer": 4,
                    "activation_shape": [4096],
                    "scores": {"r": scores},
                }
            )
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    main(
        [
            "lens-compare",
            "--records",
            str(records_path),
            "--registry",
            str(registry_path),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    boundary = next(item for item in report["comparisons"] if item["concept"] == "boundary_denial")

    assert boundary["proceed_minus_stop"] == -0.5
