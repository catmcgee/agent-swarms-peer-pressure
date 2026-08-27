from pathlib import Path

from swarmstop.config import (
    load_experiment_config,
    load_peer_boards,
    load_tasks,
    validate_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_matrix_has_full_factorial() -> None:
    config = load_experiment_config(ROOT / "configs/smoke.yaml")
    tasks = load_tasks(config.tasks_path)
    boards = load_peer_boards(config.peer_boards_path)

    validate_experiment(config, tasks, boards)

    assert len(config.conditions) == 24
    assert len(tasks) == 2
    assert len(tasks) * len(config.conditions) * len(config.seeds) == 48
