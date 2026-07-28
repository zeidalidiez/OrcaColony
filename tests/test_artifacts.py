import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts


CORPUS = (
    "Once upon a time, a small fox found a bright red ball. "
    "The fox shared the ball with a kind bird.\n<|endoftext|>\n"
    "A little boat sailed across the blue pond and came safely home. "
    "Everyone cheered for the brave little boat.\n<|endoftext|>\n"
) * 30


def test_dataset_artifacts_are_deterministic_and_pack_shifted_targets(
    tmp_path: Path,
) -> None:
    source = {
        "dataset": "test/tiny-stories",
        "revision": "0123456789abcdef",
        "license": "cdla-sharing-1.0",
    }
    first = build_dataset_artifacts(
        train_bytes=CORPUS.encode("utf-8"),
        validation_bytes=CORPUS[: len(CORPUS) // 2].encode("utf-8"),
        output_dir=tmp_path / "first",
        source=source,
        vocab_size=300,
        context_length=16,
    )
    second = build_dataset_artifacts(
        train_bytes=CORPUS.encode("utf-8"),
        validation_bytes=CORPUS[: len(CORPUS) // 2].encode("utf-8"),
        output_dir=tmp_path / "second",
        source=source,
        vocab_size=300,
        context_length=16,
    )

    assert first["tokenizer"]["sha256"] == second["tokenizer"]["sha256"]
    assert first["files"]["train.safetensors"] == second["files"]["train.safetensors"]
    assert first["files"]["validation.safetensors"] == second["files"][
        "validation.safetensors"
    ]
    for name in ("tokenizer.json", "DATASET-NOTICE.md", "manifest.json"):
        payload = (tmp_path / "first" / name).read_bytes()
        assert b"\n" in payload
        assert b"\r" not in payload
    assert first["tokenizer"]["sha256"] == hashlib.sha256(
        (tmp_path / "first" / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert first["packing"]["train_sequences"] > 0
    assert first["packing"]["validation_sequences"] > 0

    packed = load_file(str(tmp_path / "first" / "train.safetensors"))
    assert packed["input_ids"].shape[1] == 16
    assert packed["target_ids"].shape == packed["input_ids"].shape
    assert (packed["input_ids"][:, 1:] == packed["target_ids"][:, :-1]).all()

    dataset = PackedDataset.load(tmp_path / "first")
    inputs, targets = dataset.batch(cursor=0, batch_size=2, sequence_limit=10)
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64
    assert inputs.tolist() == packed["input_ids"][:2].tolist()
    assert targets.tolist() == packed["target_ids"][:2].tolist()

    admitted_train = dataset.artifact_bytes("train.safetensors")
    replacement = tmp_path / "first" / "replacement-train.safetensors"
    replacement.write_bytes(b"mutated after dataset admission")
    replacement.replace(tmp_path / "first" / "train.safetensors")
    repeated_inputs, repeated_targets = dataset.batch(
        cursor=0,
        batch_size=2,
        sequence_limit=10,
    )
    assert repeated_inputs.tolist() == inputs.tolist()
    assert repeated_targets.tolist() == targets.tolist()
    assert dataset.artifact_bytes("train.safetensors") == admitted_train
