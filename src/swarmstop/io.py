from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import ExperimentConfig, TrialResult


def completed_trial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "completed":
                completed.add(str(record["trial_id"]))
    return completed


class ResultWriter:
    def __init__(self, output_dir: str | Path, config: ExperimentConfig):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "trials.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.completed = completed_trial_ids(self.results_path)
        if not self.manifest_path.exists():
            self._write_manifest(config)

    def append(self, result: TrialResult) -> None:
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True, default=str) + "\n")
            handle.flush()
        if result.status == "completed":
            self.completed.add(result.trial_id)

    def _write_manifest(self, config: ExperimentConfig) -> None:
        payload: dict[str, Any] = {
            "created_at": datetime.now(UTC).isoformat(),
            "runner_revision": _git_revision(),
            "config": _serialize_config(config),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def iter_results(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    raw = asdict(config)
    for name in ("feasibility", "investment", "peer_norm"):
        raw[name] = [item.value for item in getattr(config, name)]
    return raw


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()
