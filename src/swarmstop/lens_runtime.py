from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .mechanistic import (
    ConceptRegistry,
    LensKind,
    MechanisticConfig,
    load_anchor_snapshots,
)
from .replay import render_snapshot


def extract_lens_records(
    *,
    config: MechanisticConfig,
    snapshots_path: str | Path,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    registry: ConceptRegistry,
) -> dict[str, Any]:
    """Capture selected residuals and score the pinned J/R lens pair on one GPU."""
    import torch
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError("lens extraction requires a CUDA GPU")
    if not config.model_revision or not config.tokenizer_revision:
        raise ValueError("model and tokenizer revisions must be pinned")

    snapshots = load_anchor_snapshots(snapshots_path)
    for snapshot in snapshots:
        if (snapshot.model_id, snapshot.model_revision) != (
            config.model_id,
            config.model_revision,
        ):
            raise ValueError(f"snapshot provenance mismatch: {snapshot.snapshot_id}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    text_module, layers, final_norm, lm_head = _resolve_text_stack(model)
    source_layers = tuple(range(31))
    if len(layers) != 32:
        raise ValueError(f"expected 32 text layers, found {len(layers)}")

    residuals, render_rows = _capture_residuals(
        snapshots=snapshots,
        tokenizer=tokenizer,
        text_module=text_module,
        layers=layers,
        source_layers=source_layers,
        device=torch.device("cuda"),
    )
    residual_path = destination / "anchor-residuals.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "source_layers": list(source_layers),
            "snapshot_ids": [snapshot.snapshot_id for snapshot in snapshots],
            "target_indices": [row["target_index"] for row in render_rows],
            "residuals": residuals,
        },
        residual_path,
    )

    concept_weights, token_registry = _concept_weights(
        tokenizer=tokenizer,
        registry=registry,
        lm_head=lm_head,
        device=torch.device("cuda"),
    )
    all_scores: dict[LensKind, Any] = {
        LensKind.LOGIT: _score_direct(
            residuals=residuals,
            final_norm=final_norm,
            concept_weights=concept_weights,
            device=torch.device("cuda"),
        )
    }
    artifact_paths: dict[str, str] = {}
    for artifact in config.artifact_files:
        local_path = hf_hub_download(
            repo_id=config.artifact_repo,
            filename=artifact.path,
            revision=config.artifact_revision,
        )
        if _sha256(Path(local_path)) != artifact.sha256:
            raise ValueError(f"downloaded lens hash mismatch: {artifact.path}")
        artifact_paths[artifact.lens.value] = local_path
        payload = torch.load(local_path, map_location="cpu", weights_only=True)
        matrices = payload.get("J")
        if not isinstance(matrices, dict) or sorted(matrices) != list(source_layers):
            raise ValueError(f"unexpected lens layer set: {artifact.path}")
        all_scores[artifact.lens] = _score_mapped(
            residuals=residuals,
            matrices=matrices,
            source_layers=source_layers,
            final_norm=final_norm,
            concept_weights=concept_weights,
            device=torch.device("cuda"),
        )
        del matrices, payload
        torch.cuda.empty_cache()

    records_path = destination / "lens-records.jsonl"
    _write_records(
        path=records_path,
        snapshots=snapshots,
        source_layers=source_layers,
        scores=all_scores,
        concept_names=registry.names,
        config=config,
        activation_ref=residual_path.name,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "lens_repo": config.artifact_repo,
        "lens_revision": config.artifact_revision,
        "lens_files": {
            artifact.lens.value: {
                "path": artifact.path,
                "sha256": artifact.sha256,
                "cached_path": artifact_paths[artifact.lens.value],
            }
            for artifact in config.artifact_files
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "snapshots": len(snapshots),
        "layers": list(source_layers),
        "concept_tokens": token_registry,
        "score": "mean concept-token logit minus mean-vocabulary logit after final norm",
        "rendered_snapshots": render_rows,
    }
    manifest_path = destination / "lens-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "snapshots": len(snapshots),
        "records": len(snapshots) * len(source_layers),
        "residuals": str(residual_path),
        "records_path": str(records_path),
        "manifest": str(manifest_path),
    }


def _capture_residuals(
    *,
    snapshots: list[Any],
    tokenizer: Any,
    text_module: Any,
    layers: Any,
    source_layers: tuple[int, ...],
    device: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    captured_rows: list[Any] = []
    render_rows: list[dict[str, Any]] = []
    text_module.eval()
    for index, snapshot in enumerate(snapshots, start=1):
        rendered = render_snapshot(snapshot, tokenizer, enable_thinking=False)
        input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
        with _record_blocks(layers, source_layers) as activations:
            with torch.inference_mode():
                text_module(input_ids=input_ids, use_cache=False)
        selected = torch.stack(
            [activations[layer][0, rendered.target_index].detach().cpu() for layer in source_layers]
        ).to(torch.bfloat16)
        captured_rows.append(selected)
        render_rows.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "token_count": rendered.token_count,
                "target_index": rendered.target_index,
                "text_sha256": hashlib.sha256(rendered.text.encode()).hexdigest(),
            }
        )
        print(json.dumps({"phase": "capture", "snapshot": index, "total": len(snapshots)}))
    return torch.stack(captured_rows), render_rows


@contextmanager
def _record_blocks(layers: Any, indices: tuple[int, ...]):
    import torch

    activations: dict[int, Any] = {}
    handles = []
    try:
        for index in indices:
            def hook(_module: Any, _inputs: Any, output: Any, *, layer: int = index) -> None:
                activations[layer] = output if torch.is_tensor(output) else output[0]

            handles.append(layers[index].register_forward_hook(hook))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _resolve_text_stack(model: Any) -> tuple[Any, Any, Any, Any]:
    candidates = ("model.language_model", "model", "language_model")
    for path in candidates:
        candidate = model
        try:
            for attribute in path.split("."):
                candidate = getattr(candidate, attribute)
        except AttributeError:
            continue
        if all(hasattr(candidate, name) for name in ("layers", "norm", "embed_tokens")):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None:
                return candidate, candidate.layers, candidate.norm, lm_head
    raise ValueError(f"could not locate Qwen text stack in {type(model).__name__}")


def _concept_weights(*, tokenizer: Any, registry: ConceptRegistry, lm_head: Any, device: Any):
    import torch

    token_registry: dict[str, list[dict[str, Any]]] = {}
    concept_ids: list[list[int]] = []
    for concept in registry.concepts:
        ids: set[int] = set()
        forms: list[dict[str, Any]] = []
        for term in concept.terms:
            for form in (term, f" {term}"):
                encoded = tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
                    forms.append({"term": term, "form": form, "token_id": int(encoded[0])})
        if not ids:
            raise ValueError(f"concept has no single-token forms: {concept.name}")
        concept_ids.append(sorted(ids))
        token_registry[concept.name] = forms

    embedding = lm_head.weight.detach()
    vocab_mean = embedding.float().mean(dim=0)
    rows = [embedding[ids].float().mean(dim=0) - vocab_mean for ids in concept_ids]
    return torch.stack(rows).to(device), token_registry


def _score_direct(*, residuals: Any, final_norm: Any, concept_weights: Any, device: Any):
    import torch

    layer_scores = []
    for layer in range(residuals.shape[1]):
        selected = residuals[:, layer].to(device)
        normalized = final_norm(selected.to(final_norm.weight.dtype)).float()
        layer_scores.append((normalized @ concept_weights.T).cpu())
    return torch.stack(layer_scores, dim=1)


def _score_mapped(
    *,
    residuals: Any,
    matrices: dict[int, Any],
    source_layers: tuple[int, ...],
    final_norm: Any,
    concept_weights: Any,
    device: Any,
):
    import torch

    layer_scores = []
    for position, layer in enumerate(source_layers):
        selected = residuals[:, position].to(device=device, dtype=torch.float32)
        matrix = matrices[layer].to(device=device, dtype=torch.float32)
        mapped = selected @ matrix.T
        normalized = final_norm(mapped.to(final_norm.weight.dtype)).float()
        layer_scores.append((normalized @ concept_weights.T).cpu())
        del selected, matrix, mapped, normalized
        print(json.dumps({"phase": "lens", "layer": layer}))
    return torch.stack(layer_scores, dim=1)


def _write_records(
    *,
    path: Path,
    snapshots: list[Any],
    source_layers: tuple[int, ...],
    scores: dict[LensKind, Any],
    concept_names: tuple[str, ...],
    config: MechanisticConfig,
    activation_ref: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for snapshot_index, snapshot in enumerate(snapshots):
            for layer_index, layer in enumerate(source_layers):
                payload = {
                    "schema_version": 1,
                    "trial_id": snapshot.trial_id,
                    "task_id": snapshot.task_id,
                    "task_family": snapshot.task_family,
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                    "tokenizer_revision": config.tokenizer_revision,
                    "seed": snapshot.seed,
                    "condition": snapshot.condition.to_dict(),
                    "board_id": snapshot.board_id,
                    "anchor": snapshot.anchor.value,
                    "layer": layer,
                    "activation_ref": (
                        f"{activation_ref}#snapshot={snapshot_index}&layer={layer_index}"
                    ),
                    "activation_dtype": "bfloat16",
                    "activation_shape": [4096],
                    "scores": {
                        lens.value: {
                            concept: float(values[snapshot_index, layer_index, concept_index])
                            for concept_index, concept in enumerate(concept_names)
                        }
                        for lens, values in scores.items()
                    },
                    "unsafe_action_probability": None,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
