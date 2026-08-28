from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mechanistic import Anchor, AnchorSnapshot, TargetMode
from .qwen_chat import render_qwen_chat


@dataclass(frozen=True)
class RenderedSnapshot:
    snapshot_id: str
    text: str
    input_ids: tuple[int, ...]
    target_index: int
    target_char_start: int | None
    enable_thinking: bool

    @property
    def token_count(self) -> int:
        return len(self.input_ids)


def render_snapshot(
    snapshot: AnchorSnapshot,
    tokenizer: Any,
    *,
    enable_thinking: bool = False,
) -> RenderedSnapshot:
    add_generation_prompt = snapshot.target_mode is TargetMode.LAST_PROMPT_TOKEN
    if snapshot.model_id.startswith("Qwen/Qwen3.5"):
        if enable_thinking:
            raise ValueError("Qwen snapshot replay must keep hidden thinking disabled")
        text = render_qwen_chat(
            tokenizer,
            list(snapshot.messages),
            list(snapshot.tools),
            add_generation_prompt=add_generation_prompt,
        )
    else:
        text = tokenizer.apply_chat_template(
            list(snapshot.messages),
            tools=list(snapshot.tools) or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _as_flat_list(encoded["input_ids"])
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    if not input_ids or len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned empty or misaligned offsets")

    target_char_start: int | None = None
    if snapshot.target_mode is TargetMode.LAST_PROMPT_TOKEN:
        target_index = len(input_ids) - 1
    else:
        target_text = snapshot.target_text or ""
        target_char_start = text.rfind(target_text)
        if target_char_start < 0:
            raise ValueError(f"target text is absent from rendered snapshot {snapshot.snapshot_id}")
        target_token = next(
            (
                index
                for index, (start, end) in enumerate(offsets)
                if end > target_char_start and start <= target_char_start
            ),
            None,
        )
        if target_token is None:
            target_token = next(
                (
                    index
                    for index, (start, end) in enumerate(offsets)
                    if start >= target_char_start and end > start
                ),
                None,
            )
        if target_token is None or target_token == 0:
            raise ValueError(
                f"cannot locate a preceding action token in snapshot {snapshot.snapshot_id}"
            )
        target_index = target_token - 1

    return RenderedSnapshot(
        snapshot_id=snapshot.snapshot_id,
        text=str(text),
        input_ids=tuple(input_ids),
        target_index=target_index,
        target_char_start=target_char_start,
        enable_thinking=enable_thinking,
    )


def validate_complete_snapshot_trials(snapshots: list[AnchorSnapshot]) -> dict[str, Any]:
    by_trial: dict[str, set[str]] = {}
    for snapshot in snapshots:
        by_trial.setdefault(snapshot.trial_id, set()).add(snapshot.anchor.value)
    expected = {anchor.value for anchor in Anchor}
    incomplete = {
        trial_id: sorted(expected - anchors)
        for trial_id, anchors in by_trial.items()
        if anchors != expected
    }
    return {
        "trials": len(by_trial),
        "snapshots": len(snapshots),
        "complete_trials": len(by_trial) - len(incomplete),
        "incomplete_trials": incomplete,
    }


def _as_flat_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("snapshot tokenizer returned an unexpected batch")
        value = value[0]
    return [int(item) for item in value]
