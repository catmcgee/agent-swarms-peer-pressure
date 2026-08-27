#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "upstreams"

REPOSITORIES = {
    "agentabstain": (
        "https://github.com/AntiQuality/agentabstain.git",
        "f581249704b26804e28a39e37396f1be00b71a4d",
    ),
    "authoritybench": (
        "https://github.com/yazcaleb/can-is-not-may.git",
        "da8a0ce8c779da067ccf6caa5dd311c1ff443960",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="store_true", help="fetch pinned source repositories")
    parser.add_argument("--dataset", action="store_true", help="fetch the AgentAbstain dataset")
    args = parser.parse_args()
    if not args.source and not args.dataset:
        parser.error("choose --source, --dataset, or both")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if args.source:
        for name, (url, revision) in REPOSITORIES.items():
            fetch_repository(name, url, revision)
    if args.dataset:
        fetch_dataset()


def fetch_repository(name: str, url: str, revision: str) -> None:
    destination = DESTINATION / name
    if not destination.exists():
        subprocess.run(["git", "clone", url, str(destination)], check=True)
    subprocess.run(["git", "fetch", "origin", revision], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=destination, check=True)
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != revision:
        raise RuntimeError(f"revision mismatch for {name}: expected {revision}, got {actual}")
    print(f"{name}: {actual}")


def fetch_dataset() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("install dataset support with: pip install -e '.[dataset]'") from exc
    destination = DESTINATION / "agentabstain-data"
    path = snapshot_download(
        "antiquality/agentabstain",
        repo_type="dataset",
        local_dir=destination,
    )
    print(f"agentabstain-data: {path}")


if __name__ == "__main__":
    main()
