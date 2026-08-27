from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from .schema import PeerNorm, TrialCondition


class Anchor(StrEnum):
    PRE_AUTHORITY = "pre_authority"
    POST_RECOGNITION = "post_recognition"
    POST_PEER = "post_peer"
    ACTION_DECISION = "action_decision"


class LensKind(StrEnum):
    J = "j"
    R = "r"
    LOGIT = "logit"


@dataclass(frozen=True)
class ArtifactFile:
    lens: LensKind
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Concept:
    name: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class ConceptRegistry:
    version: int
    concepts: tuple[Concept, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(concept.name for concept in self.concepts)


@dataclass(frozen=True)
class MechanisticConfig:
    name: str
    output_dir: str
    registry_path: str
    model_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    artifact_repo: str
    artifact_revision: str
    artifact_subdir: str
    artifact_files: tuple[ArtifactFile, ...]
    task_families: int
    seeds: tuple[int, ...]
    feasibility: tuple[str, ...]
    peer_norm: tuple[PeerNorm, ...]
    investment: tuple[str, ...]
    budget_rounds: tuple[int, ...]
    anchors: tuple[Anchor, ...]
    activation_dtype: str

    @property
    def planned_trajectories(self) -> int:
        return (
            self.task_families
            * len(self.seeds)
            * len(self.feasibility)
            * len(self.peer_norm)
            * len(self.investment)
            * len(self.budget_rounds)
        )

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        values = {
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }
        return tuple(name for name, value in values.items() if not value)

    @property
    def ready(self) -> bool:
        return not self.unresolved_fields


@dataclass(frozen=True)
class LensAnchorRecord:
    schema_version: int
    trial_id: str
    task_id: str
    task_family: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    seed: int
    condition: TrialCondition
    board_id: str | None
    anchor: Anchor
    layer: int
    activation_ref: str | None
    activation_dtype: str | None
    activation_shape: tuple[int, ...]
    scores: dict[LensKind, dict[str, float]]
    unsafe_action_probability: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported mechanistic record schema: {self.schema_version}")
        if not self.trial_id or not self.task_id or not self.task_family:
            raise ValueError("trial_id, task_id, and task_family are required")
        if not self.model_id or not self.model_revision or not self.tokenizer_revision:
            raise ValueError("model and tokenizer provenance are required")
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if any(size < 1 for size in self.activation_shape):
            raise ValueError("activation_shape values must be positive")
        if not self.scores:
            raise ValueError("at least one lens score mapping is required")
        for lens, concept_scores in self.scores.items():
            if not concept_scores:
                raise ValueError(f"lens {lens.value} has no concept scores")
            for concept, score in concept_scores.items():
                if not concept or not math.isfinite(score):
                    raise ValueError("concept names must be nonempty and scores finite")
        probability = self.unsafe_action_probability
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError("unsafe_action_probability must be between zero and one")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LensAnchorRecord:
        raw_scores = value.get("scores") or {}
        scores = {
            LensKind(str(lens)): {
                str(concept): float(score) for concept, score in concept_scores.items()
            }
            for lens, concept_scores in raw_scores.items()
        }
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            trial_id=str(value.get("trial_id", "")),
            task_id=str(value.get("task_id", "")),
            task_family=str(value.get("task_family", "")),
            model_id=str(value.get("model_id", "")),
            model_revision=str(value.get("model_revision", "")),
            tokenizer_revision=str(value.get("tokenizer_revision", "")),
            seed=int(value.get("seed", 0)),
            condition=TrialCondition.from_dict(dict(value.get("condition") or {})),
            board_id=value.get("board_id"),
            anchor=Anchor(str(value["anchor"])),
            layer=int(value["layer"]),
            activation_ref=value.get("activation_ref"),
            activation_dtype=value.get("activation_dtype"),
            activation_shape=tuple(int(item) for item in value.get("activation_shape", [])),
            scores=scores,
            unsafe_action_probability=(
                float(value["unsafe_action_probability"])
                if value.get("unsafe_action_probability") is not None
                else None
            ),
        )


def load_concept_registry(path: str | Path) -> ConceptRegistry:
    raw = _load_yaml(Path(path))
    if not isinstance(raw, dict):
        raise ValueError("concept registry must be a mapping")
    raw_concepts = raw.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError("concept registry must contain a nonempty concepts list")

    concepts: list[Concept] = []
    for item in raw_concepts:
        if not isinstance(item, dict):
            raise ValueError("each concept must be a mapping")
        name = str(item.get("name", "")).strip()
        terms = tuple(str(term).strip() for term in item.get("terms", []) if str(term).strip())
        if not name or not terms:
            raise ValueError("each concept requires a name and at least one term")
        if len(terms) != len(set(terms)):
            raise ValueError(f"concept {name} contains duplicate terms")
        concepts.append(Concept(name=name, terms=terms))

    names = [concept.name for concept in concepts]
    if len(names) != len(set(names)):
        raise ValueError("concept names must be unique")
    return ConceptRegistry(version=int(raw.get("version", 1)), concepts=tuple(concepts))


def load_mechanistic_config(path: str | Path) -> MechanisticConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    if not isinstance(raw, dict):
        raise ValueError("mechanistic config must be a mapping")
    base = config_path.parent
    model = raw.get("model") or {}
    artifact = raw.get("lens_artifact") or {}
    artifact_files = artifact.get("files") or {}
    factors = raw.get("factors") or {}

    def resolved(value: str) -> str:
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

    config = MechanisticConfig(
        name=str(raw.get("name", config_path.stem)),
        output_dir=resolved(str(raw.get("output_dir", "../results/jr-lens-discovery"))),
        registry_path=resolved(str(raw["concept_registry_path"])),
        model_id=str(model.get("id", "")),
        model_revision=_optional_string(model.get("revision")),
        tokenizer_revision=_optional_string(model.get("tokenizer_revision")),
        artifact_repo=str(artifact.get("repo", "")),
        artifact_revision=str(artifact.get("revision", "")),
        artifact_subdir=str(artifact.get("subdir", "")),
        artifact_files=tuple(
            ArtifactFile(
                lens=LensKind(str(lens)),
                path=str(item.get("path", "")),
                size=int(item.get("size", 0)),
                sha256=str(item.get("sha256", "")),
            )
            for lens, item in artifact_files.items()
        ),
        task_families=int(raw.get("task_families", 0)),
        seeds=tuple(int(seed) for seed in raw.get("seeds", [])),
        feasibility=tuple(str(item) for item in factors.get("feasibility", [])),
        peer_norm=tuple(PeerNorm(str(item)) for item in factors.get("peer_norm", [])),
        investment=tuple(str(item) for item in factors.get("investment", [])),
        budget_rounds=tuple(int(item) for item in factors.get("budget_rounds", [])),
        anchors=tuple(Anchor(str(item)) for item in raw.get("anchors", [])),
        activation_dtype=str(raw.get("activation_dtype", "bfloat16")),
    )
    validate_mechanistic_config(config)
    return config


def validate_mechanistic_config(config: MechanisticConfig) -> None:
    if not config.model_id:
        raise ValueError("mechanistic model ID is required")
    if not config.artifact_repo or not config.artifact_revision or not config.artifact_subdir:
        raise ValueError("lens artifact repo, revision, and subdir are required")
    artifact_lenses = {item.lens for item in config.artifact_files}
    if artifact_lenses != {LensKind.J, LensKind.R} or len(config.artifact_files) != 2:
        raise ValueError("exactly one J-lens and one R-lens artifact file are required")
    for item in config.artifact_files:
        if not item.path or item.size < 1:
            raise ValueError("artifact files require a path and positive size")
        if len(item.sha256) != 64 or any(char not in "0123456789abcdef" for char in item.sha256):
            raise ValueError("artifact file sha256 must be 64 lowercase hexadecimal characters")
    if config.task_families < 1 or not config.seeds:
        raise ValueError("task_families and seeds must be nonempty")
    if not all((config.feasibility, config.peer_norm, config.investment, config.budget_rounds)):
        raise ValueError("all mechanistic factors must be nonempty")
    required = set(Anchor)
    if set(config.anchors) != required or len(config.anchors) != len(required):
        raise ValueError("mechanistic config must contain each of the four anchors exactly once")
    if any(rounds < 1 for rounds in config.budget_rounds):
        raise ValueError("budget rounds must be positive")


def load_lens_records(
    path: str | Path, registry: ConceptRegistry | None = None
) -> list[LensAnchorRecord]:
    records: list[LensAnchorRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = LensAnchorRecord.from_dict(json.loads(line))
                if registry is not None:
                    _validate_record_concepts(record, registry)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                message = f"invalid mechanistic record on line {line_number}: {exc}"
                raise ValueError(message) from exc
            records.append(record)
    if not records:
        raise ValueError("mechanistic record file is empty")
    _validate_record_provenance(records)
    return records


def peer_delta_contrasts(
    records: list[LensAnchorRecord],
    *,
    from_anchor: Anchor = Anchor.POST_RECOGNITION,
    to_anchor: Anchor = Anchor.POST_PEER,
) -> dict[str, Any]:
    values: dict[tuple[str, int, LensKind, str, Anchor], float] = {}
    norms: dict[str, PeerNorm] = {}
    for record in records:
        norms.setdefault(record.trial_id, record.condition.peer_norm)
        if norms[record.trial_id] != record.condition.peer_norm:
            raise ValueError(f"trial {record.trial_id} has inconsistent peer norms")
        if record.anchor not in {from_anchor, to_anchor}:
            continue
        for lens, concept_scores in record.scores.items():
            for concept, score in concept_scores.items():
                key = (record.trial_id, record.layer, lens, concept, record.anchor)
                if key in values:
                    raise ValueError(f"duplicate score record for {key}")
                values[key] = score

    grouped: dict[tuple[int, LensKind, str, PeerNorm], list[float]] = defaultdict(list)
    candidate_keys = {
        (trial_id, layer, lens, concept) for trial_id, layer, lens, concept, _anchor in values
    }
    incomplete_pairs = 0
    for trial_id, layer, lens, concept in candidate_keys:
        norm = norms[trial_id]
        if norm not in {PeerNorm.STOP, PeerNorm.PROCEED}:
            continue
        before = values.get((trial_id, layer, lens, concept, from_anchor))
        after = values.get((trial_id, layer, lens, concept, to_anchor))
        if before is None or after is None:
            incomplete_pairs += 1
            continue
        grouped[(layer, lens, concept, norm)].append(after - before)

    comparisons: list[dict[str, Any]] = []
    comparison_keys = sorted(
        {(layer, lens, concept) for layer, lens, concept, _norm in grouped},
        key=lambda item: (item[0], item[1].value, item[2]),
    )
    for layer, lens, concept in comparison_keys:
        stop = grouped.get((layer, lens, concept, PeerNorm.STOP), [])
        proceed = grouped.get((layer, lens, concept, PeerNorm.PROCEED), [])
        if not stop or not proceed:
            continue
        stop_mean = fmean(stop)
        proceed_mean = fmean(proceed)
        comparisons.append(
            {
                "layer": layer,
                "lens": lens.value,
                "concept": concept,
                "n_stop": len(stop),
                "n_proceed": len(proceed),
                "mean_delta_stop": stop_mean,
                "mean_delta_proceed": proceed_mean,
                "proceed_minus_stop": proceed_mean - stop_mean,
            }
        )

    return {
        "from_anchor": from_anchor.value,
        "to_anchor": to_anchor.value,
        "incomplete_pairs": incomplete_pairs,
        "comparisons": comparisons,
    }


def _validate_record_concepts(record: LensAnchorRecord, registry: ConceptRegistry) -> None:
    expected = set(registry.names)
    for lens, scores in record.scores.items():
        observed = set(scores)
        if observed != expected:
            missing = sorted(expected - observed)
            unknown = sorted(observed - expected)
            raise ValueError(
                f"lens {lens.value} concept mismatch; missing={missing}, unknown={unknown}"
            )


def _validate_record_provenance(records: list[LensAnchorRecord]) -> None:
    provenance = {
        (record.model_id, record.model_revision, record.tokenizer_revision) for record in records
    }
    if len(provenance) != 1:
        raise ValueError("mechanistic record file mixes model or tokenizer revisions")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
