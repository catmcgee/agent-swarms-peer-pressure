from __future__ import annotations

import collections
import io
import json
import pickle
import struct
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from typing import Any

from .mechanistic import ArtifactFile, LensKind, MechanisticConfig

RangeFetcher = Callable[[str, int, int], bytes]


def inspect_lens_pair(config: MechanisticConfig) -> dict[str, Any]:
    tree = _fetch_json(_artifact_tree_url(config))
    if not isinstance(tree, list):
        raise ValueError("artifact tree response must be a list")
    remote_files = {str(item.get("path")): item for item in tree if item.get("type") == "file"}

    reports: dict[str, dict[str, Any]] = {}
    for expected in config.artifact_files:
        remote = remote_files.get(expected.path)
        if remote is None:
            raise ValueError(f"artifact file is missing: {expected.path}")
        _validate_remote_file(expected, remote)
        url = _artifact_download_url(config, expected.path)
        metadata = read_remote_torch_metadata(url, expected.size)
        reports[expected.lens.value] = {
            "path": expected.path,
            "size": expected.size,
            "sha256": expected.sha256,
            **metadata,
        }

    matched_fields = _validate_matched_pair(config, reports)
    model_report = _inspect_model_config(config, reports[LensKind.J.value])
    return {
        "artifact_repo": config.artifact_repo,
        "artifact_revision": config.artifact_revision,
        "files": reports,
        "matched_fields": matched_fields,
        "model": model_report,
        "verified": True,
    }


def read_remote_torch_metadata(
    url: str,
    size: int,
    *,
    fetch: RangeFetcher | None = None,
) -> dict[str, Any]:
    range_fetch = fetch or _fetch_range
    member = _read_remote_zip_member(url, size, "data.pkl", range_fetch)
    payload = _MetadataUnpickler(io.BytesIO(member)).load()
    if not isinstance(payload, dict):
        raise ValueError("torch artifact metadata root must be a mapping")
    tensors = payload.get("J")
    if not isinstance(tensors, dict):
        raise ValueError("torch artifact does not contain a J tensor mapping")
    tensor_shapes = sorted(
        {tuple(int(value) for value in tensor.get("size", ())) for tensor in tensors.values()}
    )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("torch artifact does not contain provenance")
    try:
        recipe = json.loads(str(provenance["config_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact provenance has invalid config_json") from exc
    return {
        "d_model": int(payload["d_model"]),
        "n_prompts": int(payload["n_prompts"]),
        "source_layers": [int(layer) for layer in payload["source_layers"]],
        "tensor_count": len(tensors),
        "tensor_shapes": [list(shape) for shape in tensor_shapes],
        "provenance": {**provenance, "config_json": recipe},
    }


def _read_remote_zip_member(
    url: str,
    size: int,
    suffix: str,
    fetch: RangeFetcher,
) -> bytes:
    tail_size = min(size, 131_072)
    tail_start = size - tail_size
    tail = fetch(url, tail_start, size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    if len(tail) < eocd_offset + 22:
        raise ValueError("truncated ZIP end-of-central-directory record")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    central_size, central_offset = int(eocd[5]), int(eocd[6])
    central = fetch(url, central_offset, central_offset + central_size - 1)
    entry = _find_central_entry(central, suffix)

    local_offset = entry["local_offset"]
    header = fetch(url, local_offset, local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local-file header")
    name_length, extra_length = struct.unpack_from("<2H", header, 26)
    data_offset = local_offset + 30 + name_length + extra_length
    compressed = fetch(url, data_offset, data_offset + entry["compressed_size"] - 1)
    if entry["method"] == 0:
        result = compressed
    elif entry["method"] == 8:
        result = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression method: {entry['method']}")
    if len(result) != entry["uncompressed_size"]:
        raise ValueError("ZIP member size does not match central directory")
    return result


def _find_central_entry(central: bytes, suffix: str) -> dict[str, Any]:
    position = 0
    matches: list[dict[str, Any]] = []
    while position < len(central):
        if central[position : position + 4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory entry")
        method = struct.unpack_from("<H", central, position + 10)[0]
        compressed_size = struct.unpack_from("<L", central, position + 20)[0]
        uncompressed_size = struct.unpack_from("<L", central, position + 24)[0]
        name_length, extra_length, comment_length = struct.unpack_from(
            "<3H", central, position + 28
        )
        local_offset = struct.unpack_from("<L", central, position + 42)[0]
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode("utf-8")
        if name == suffix or name.endswith(f"/{suffix}"):
            matches.append(
                {
                    "name": name,
                    "method": method,
                    "compressed_size": compressed_size,
                    "uncompressed_size": uncompressed_size,
                    "local_offset": local_offset,
                }
            )
        position = name_start + name_length + extra_length + comment_length
    if len(matches) != 1:
        raise ValueError(f"expected one ZIP member ending in {suffix}, found {len(matches)}")
    return matches[0]


def _validate_remote_file(expected: ArtifactFile, remote: dict[str, Any]) -> None:
    if int(remote.get("size", -1)) != expected.size:
        raise ValueError(f"artifact size mismatch for {expected.path}")
    lfs = remote.get("lfs") or {}
    if str(lfs.get("oid", "")) != expected.sha256:
        raise ValueError(f"artifact sha256 mismatch for {expected.path}")


def _validate_matched_pair(
    config: MechanisticConfig, reports: dict[str, dict[str, Any]]
) -> list[str]:
    j_report = reports[LensKind.J.value]
    r_report = reports[LensKind.R.value]
    fields = (
        "model_id",
        "dataset_id",
        "target_layer",
        "t_max",
        "n_prompts",
        "skip_first",
        "weighting",
        "corpus_mode",
    )
    for field in fields:
        if j_report["provenance"].get(field) != r_report["provenance"].get(field):
            raise ValueError(f"J/R provenance mismatch for {field}")
    for field in ("d_model", "n_prompts", "source_layers", "tensor_shapes"):
        if j_report[field] != r_report[field]:
            raise ValueError(f"J/R artifact mismatch for {field}")
    if j_report["provenance"].get("model_id") != config.model_id:
        raise ValueError("artifact model ID does not match mechanistic config")
    if j_report["provenance"]["config_json"].get("estimator") != "standard":
        raise ValueError("J-lens artifact does not use the standard estimator")
    if r_report["provenance"]["config_json"].get("estimator") != "relp":
        raise ValueError("R-lens artifact does not use the RelP estimator")
    return list(fields) + ["d_model", "source_layers", "tensor_shapes"]


def _inspect_model_config(config: MechanisticConfig, lens_report: dict[str, Any]) -> dict[str, Any]:
    if not config.model_revision or not config.tokenizer_revision:
        raise ValueError("model and tokenizer revisions must be pinned before inspection")
    model_url = _model_file_url(config.model_id, config.model_revision, "config.json")
    model_config = _fetch_json(model_url)
    hidden_size, layers = _language_model_dimensions(model_config)
    if hidden_size != lens_report["d_model"]:
        raise ValueError("model hidden size does not match lens d_model")
    target_layer = int(lens_report["provenance"]["target_layer"])
    if target_layer != layers - 2:
        raise ValueError("lens target layer is not the model's penultimate layer")
    return {
        "id": config.model_id,
        "revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "hidden_size": hidden_size,
        "num_hidden_layers": layers,
        "target_layer": target_layer,
    }


def _language_model_dimensions(model_config: dict[str, Any]) -> tuple[int, int]:
    text_config = model_config.get("text_config", model_config)
    return int(text_config["hidden_size"]), int(text_config["num_hidden_layers"])


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_metadata
        if (module, name) == ("torch", "HalfStorage"):
            return "HalfStorage"
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")

    def persistent_load(self, value: Any) -> dict[str, Any]:
        return {"persistent_id": value}


def _rebuild_tensor_metadata(
    storage: Any,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    _requires_grad: bool,
    _hooks: Any,
) -> dict[str, Any]:
    return {
        "storage": storage,
        "offset": offset,
        "size": tuple(size),
        "stride": tuple(stride),
    }


def _fetch_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError("invalid byte range")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers.get("Content-Range"):
            raise ValueError("remote server did not honor the byte-range request")
        expected = end - start + 1
        payload = response.read(expected + 1)
    if len(payload) != expected:
        raise ValueError("byte-range response length mismatch")
    return payload


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _artifact_tree_url(config: MechanisticConfig) -> str:
    repo = urllib.parse.quote(config.artifact_repo, safe="/")
    revision = urllib.parse.quote(config.artifact_revision, safe="")
    subdir = urllib.parse.quote(config.artifact_subdir, safe="/")
    return (
        f"https://huggingface.co/api/models/{repo}/tree/{revision}/{subdir}"
        "?recursive=true&expand=false"
    )


def _artifact_download_url(config: MechanisticConfig, path: str) -> str:
    repo = urllib.parse.quote(config.artifact_repo, safe="/")
    revision = urllib.parse.quote(config.artifact_revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{encoded_path}"


def _model_file_url(model_id: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(model_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{encoded_revision}/{encoded_path}"
