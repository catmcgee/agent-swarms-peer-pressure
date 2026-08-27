import io
import json
import pickle
import zipfile

from swarmstop.artifacts import _language_model_dimensions, read_remote_torch_metadata


def test_reads_torch_metadata_with_range_fetches_only() -> None:
    payload = {
        "J": {0: {"size": (4, 4)}},
        "n_prompts": 25,
        "source_layers": [0],
        "d_model": 4,
        "provenance": {
            "model_id": "model-1",
            "target_layer": 0,
            "config_json": json.dumps({"estimator": "standard"}),
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("lens/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr("lens/data/0", b"\0" * 1_000_000)
    artifact = buffer.getvalue()
    fetched_bytes = 0

    def fetch(_url: str, start: int, end: int) -> bytes:
        nonlocal fetched_bytes
        result = artifact[start : end + 1]
        fetched_bytes += len(result)
        return result

    report = read_remote_torch_metadata(
        "https://example.invalid/lens.pt", len(artifact), fetch=fetch
    )

    assert report["d_model"] == 4
    assert report["tensor_shapes"] == [[4, 4]]
    assert report["provenance"]["config_json"] == {"estimator": "standard"}
    assert fetched_bytes < len(artifact)


def test_reads_language_dimensions_from_nested_multimodal_config() -> None:
    config = {"text_config": {"hidden_size": 4096, "num_hidden_layers": 32}}

    assert _language_model_dimensions(config) == (4096, 32)
