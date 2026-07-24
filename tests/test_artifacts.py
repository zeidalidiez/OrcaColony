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
