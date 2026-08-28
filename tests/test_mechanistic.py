import json
from pathlib import Path

import pytest

from swarmstop.mechanistic import (
    Anchor,
    AnchorSnapshot,
    LensAnchorRecord,
    SnapshotWriter,
    TargetMode,
    load_anchor_snapshots,
    load_concept_registry,
    load_mechanistic_config,
    peer_delta_contrasts,
)

ROOT = Path(__file__).resolve().parents[1]


def _record(trial_id: str, norm: str, anchor: str, boundary: float) -> LensAnchorRecord:
    return LensAnchorRecord.from_dict(
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
            "scores": {"r": {"boundary_denial": boundary}},
        }
    )


def test_discovery_config_has_planned_size_and_pinned_provenance() -> None:
    config = load_mechanistic_config(ROOT / "configs/jr_lens.yaml")
    registry = load_concept_registry(config.registry_path)

    assert config.planned_trajectories == 216
    assert config.ready is True
    assert config.unresolved_fields == ()
    assert {item.lens.value for item in config.artifact_files} == {"j", "r"}
    assert "boundary_denial" in registry.names


def test_peer_delta_contrast_is_within_trial_then_between_norms() -> None:
    records = [
        _record("stop-1", "stop", "post_recognition", 0.8),
        _record("stop-1", "stop", "post_peer", 0.9),
        _record("proceed-1", "proceed", "post_recognition", 0.8),
        _record("proceed-1", "proceed", "post_peer", 0.4),
    ]

    result = peer_delta_contrasts(records)

    assert result["incomplete_pairs"] == 0
    assert result["comparisons"] == [
        {
            "layer": 4,
            "lens": "r",
            "concept": "boundary_denial",
            "n_stop": 1,
            "n_proceed": 1,
            "mean_delta_stop": pytest.approx(0.1),
            "mean_delta_proceed": pytest.approx(-0.4),
            "proceed_minus_stop": pytest.approx(-0.5),
        }
    ]


def test_record_requires_complete_registry_when_loaded(tmp_path: Path) -> None:
    record = _record("stop-1", "stop", "post_peer", 0.9)
    record_path = tmp_path / "records.jsonl"
    payload = {
        "schema_version": record.schema_version,
        "trial_id": record.trial_id,
        "task_id": record.task_id,
        "task_family": record.task_family,
        "model_id": record.model_id,
        "model_revision": record.model_revision,
        "tokenizer_revision": record.tokenizer_revision,
        "seed": record.seed,
        "condition": record.condition.to_dict(),
        "anchor": record.anchor.value,
        "layer": record.layer,
        "activation_shape": list(record.activation_shape),
        "scores": {"r": {"boundary_denial": 0.9}},
    }
    record_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    registry = load_concept_registry(ROOT / "configs/concepts/jr_lens.yaml")

    from swarmstop.mechanistic import load_lens_records

    with pytest.raises(ValueError, match="concept mismatch"):
        load_lens_records(record_path, registry)


def test_anchor_enum_matches_protocol_names() -> None:
    assert tuple(Anchor) == (
        Anchor.PRE_AUTHORITY,
        Anchor.POST_RECOGNITION,
        Anchor.POST_PEER,
        Anchor.ACTION_DECISION,
    )


def test_snapshot_writer_is_resumable_and_rejects_changed_context(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.jsonl"
    snapshot = AnchorSnapshot(
        schema_version=1,
        trial_id="trial-1",
        task_id="task-1",
        task_family="family-1",
        model_id="model-1",
        model_revision="revision-1",
        seed=1,
        condition=_record("trial-1", "stop", "post_peer", 0.9).condition,
        board_id="board-1",
        anchor=Anchor.POST_PEER,
        messages=({"role": "user", "content": "test"},),
        tools=(),
        target_mode=TargetMode.LAST_PROMPT_TOKEN,
    )
    writer = SnapshotWriter(path)

    writer.capture(snapshot)
    writer.capture(snapshot)
    resumed = SnapshotWriter(path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert resumed.has_complete_trial("trial-1") is False

    changed = AnchorSnapshot(
        **{**snapshot.__dict__, "messages": ({"role": "user", "content": "changed"},)}
    )
    with pytest.raises(ValueError, match="snapshot changed"):
        resumed.capture(changed)

    loaded = load_anchor_snapshots(path)
    assert loaded == [snapshot]
