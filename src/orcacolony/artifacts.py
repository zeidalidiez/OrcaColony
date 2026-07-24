from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


END_OF_STORY = "<|endoftext|>"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_SOURCE = {
    "dataset": "roneneldan/TinyStories",
    "revision": TINYSTORIES_REVISION,
    "license": "cdla-sharing-1.0",
    "license_url": "https://cdla.dev/sharing-1-0/",
    "dataset_card": "https://huggingface.co/datasets/roneneldan/TinyStories",
    "train": {
        "file": "TinyStories-train.txt",
        "size": 1_924_281_556,
        "sha256": "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
    },
    "validation": {
        "file": "TinyStories-valid.txt",
        "size": 19_447_282,
        "sha256": "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
    },
}


@dataclass(frozen=True)
class PackedDataset:
    root: Path
    manifest: Mapping[str, object]
    revision: str
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor

    @classmethod
    def load(cls, root: str | Path) -> PackedDataset:
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("format") != "orcacolony_dataset_artifacts_v1":
            raise ValueError("unsupported dataset artifact format")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("dataset artifact file manifest is missing")
        for filename, expected_sha256 in files.items():
            path = root / filename
            if not path.is_file() or _sha256_file(path) != expected_sha256:
                raise ValueError(f"dataset artifact digest mismatch: {filename}")

        train = load_safetensors_file(str(root / "train.safetensors"))
        validation = load_safetensors_file(str(root / "validation.safetensors"))
        train_inputs, train_targets = _validate_packed_split(train, "train")
        validation_inputs, validation_targets = _validate_packed_split(
            validation, "validation"
        )
        context_length = int(manifest["packing"]["context_length"])
        if (
            train_inputs.shape[1] != context_length
            or validation_inputs.shape[1] != context_length
        ):
            raise ValueError("packed dataset context length does not match manifest")
        if train_inputs.shape[0] != int(manifest["packing"]["train_sequences"]):
            raise ValueError("packed training sequence count does not match manifest")
        if validation_inputs.shape[0] != int(
            manifest["packing"]["validation_sequences"]
        ):
            raise ValueError("packed validation sequence count does not match manifest")
        return cls(
            root=root,
            manifest=manifest,
            revision=_sha256_bytes(manifest_bytes),
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
        )

    def batch(
        self,
        *,
        cursor: int,
        batch_size: int,
        sequence_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0 or sequence_limit <= 0:
            raise ValueError("dataset batch dimensions must be positive")
        if sequence_limit > self.train_inputs.shape[0]:
            raise ValueError("campaign dataset sequence limit exceeds packed data")
        indices = torch.tensor(
            [(cursor + offset) % sequence_limit for offset in range(batch_size)],
            dtype=torch.int64,
        )
        return (
            self.train_inputs.index_select(0, indices).to(torch.int64),
            self.train_targets.index_select(0, indices).to(torch.int64),
        )

    def validation_batch(
        self,
        *,
        cursor: int,
        batch_size: int,
        sequence_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0 or sequence_limit <= 0:
            raise ValueError("validation batch dimensions must be positive")
        if sequence_limit > self.validation_inputs.shape[0]:
            raise ValueError("evaluation sequence limit exceeds packed validation data")
        indices = torch.tensor(
            [(cursor + offset) % sequence_limit for offset in range(batch_size)],
            dtype=torch.int64,
        )
        return (
            self.validation_inputs.index_select(0, indices).to(torch.int64),
            self.validation_targets.index_select(0, indices).to(torch.int64),
        )


def _validate_packed_split(
    tensors: Mapping[str, torch.Tensor],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(tensors) != {"input_ids", "target_ids"}:
        raise ValueError(f"packed {split} tensors have unexpected names")
    inputs = tensors["input_ids"]
    targets = tensors["target_ids"]
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError(f"packed {split} tensors have invalid shapes")
    if inputs.dtype != torch.int32 or targets.dtype != torch.int32:
        raise ValueError(f"packed {split} tensors must use int32")
    if not torch.equal(inputs[:, 1:], targets[:, :-1]):
        raise ValueError(f"packed {split} targets are not shifted inputs")
    return inputs, targets


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_stories(payload: bytes) -> tuple[list[str], bytes]:
    text = payload.decode("utf-8")
    marker_end = text.rfind(END_OF_STORY)
    if marker_end < 0:
        raise ValueError("source subset does not contain a complete story")
    used_text = text[: marker_end + len(END_OF_STORY)]
    stories = [story.strip() for story in used_text.split(END_OF_STORY) if story.strip()]
    if not stories:
        raise ValueError("source subset does not contain usable stories")
    return stories, used_text.encode("utf-8")


def _train_tokenizer(stories: list[str], vocab_size: int) -> Tokenizer:
    if vocab_size < 260:
        raise ValueError("byte-level vocabulary must contain at least 260 entries")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(stories, trainer=trainer)
    return tokenizer


def _pack_stories(
    tokenizer: Tokenizer,
    stories: list[str],
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if context_length < 2:
        raise ValueError("context length must be at least two")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer special-token IDs are missing")
    token_ids: list[int] = []
    for story in stories:
        token_ids.append(bos_id)
        token_ids.extend(tokenizer.encode(story).ids)
        token_ids.append(eos_id)

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(token_ids) - context_length, context_length):
        window = token_ids[start : start + context_length + 1]
        if len(window) != context_length + 1:
            break
        inputs.append(window[:-1])
        targets.append(window[1:])
    if not inputs:
        raise ValueError("tokenized corpus is too small for one packed sequence")
    return (
        torch.tensor(inputs, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.int32),
        len(token_ids),
    )


def build_dataset_artifacts(
    *,
    train_bytes: bytes,
    validation_bytes: bytes,
    output_dir: str | Path,
    source: Mapping[str, object],
    vocab_size: int,
    context_length: int,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"dataset artifact directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stories, train_used = _complete_stories(train_bytes)
    validation_stories, validation_used = _complete_stories(validation_bytes)
    tokenizer = _train_tokenizer(train_stories, vocab_size)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_path.write_text(tokenizer.to_str(pretty=True) + "\n", encoding="utf-8")

    train_inputs, train_targets, train_tokens = _pack_stories(
        tokenizer, train_stories, context_length
    )
    validation_inputs, validation_targets, validation_tokens = _pack_stories(
        tokenizer, validation_stories, context_length
    )
    train_path = output_dir / "train.safetensors"
    validation_path = output_dir / "validation.safetensors"
    save_safetensors_file(
        {"input_ids": train_inputs, "target_ids": train_targets},
        str(train_path),
    )
    save_safetensors_file(
        {"input_ids": validation_inputs, "target_ids": validation_targets},
        str(validation_path),
    )

    notice_path = output_dir / "DATASET-NOTICE.md"
    notice_path.write_text(
        "# Dataset notice\n\n"
        f"Source: `{source.get('dataset', 'unknown')}`\n\n"
        f"Revision: `{source.get('revision', 'unknown')}`\n\n"
        f"License: `{source.get('license', 'unknown')}`\n\n"
        f"License URL: {source.get('license_url', 'not supplied')}\n\n"
        "Changes: OrcaColony selected fixed byte prefixes, removed any trailing "
        "incomplete story, trained a byte-level BPE tokenizer on the training "
        "subset, encoded the text, and packed shifted input/target tensors. "
        "These files are modified and rearranged data, not the original raw files.\n",
        encoding="utf-8",
    )

    files = {
        path.name: _sha256_file(path)
        for path in (tokenizer_path, train_path, validation_path, notice_path)
    }
    special_token_ids = {
        token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
    }
    manifest: dict[str, object] = {
        "format": "orcacolony_dataset_artifacts_v1",
        "source": dict(source),
        "subsets": {
            "train": {
                "downloaded_bytes": len(train_bytes),
                "download_sha256": _sha256_bytes(train_bytes),
                "used_bytes": len(train_used),
                "used_sha256": _sha256_bytes(train_used),
                "stories": len(train_stories),
            },
            "validation": {
                "downloaded_bytes": len(validation_bytes),
                "download_sha256": _sha256_bytes(validation_bytes),
                "used_bytes": len(validation_used),
                "used_sha256": _sha256_bytes(validation_used),
                "stories": len(validation_stories),
            },
        },
        "tokenizer": {
            "format": "byte_level_bpe",
            "requested_vocab_size": vocab_size,
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": special_token_ids,
            "sha256": files[tokenizer_path.name],
        },
        "packing": {
            "context_length": context_length,
            "stride": context_length,
            "dtype": "int32",
            "train_tokens": train_tokens,
            "validation_tokens": validation_tokens,
            "train_sequences": train_inputs.shape[0],
            "validation_sequences": validation_inputs.shape[0],
        },
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )
    return manifest


def _download_prefix(filename: str, byte_count: int) -> bytes:
    if byte_count <= 0:
        raise ValueError("download byte count must be positive")
    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
        f"{TINYSTORIES_REVISION}/{filename}"
    )
    request = Request(
        url,
        headers={
            "Range": f"bytes=0-{byte_count - 1}",
            "User-Agent": "OrcaColony/0.1 dataset-artifact-builder",
        },
    )
    with urlopen(request, timeout=120) as response:
        payload = response.read(byte_count + 1)
    if len(payload) != byte_count:
        raise ValueError(
            f"source returned {len(payload)} bytes instead of requested {byte_count}"
        )
    return payload


def build_tinystories_subset(
    output_dir: str | Path,
    *,
    train_bytes: int,
    validation_bytes: int,
    vocab_size: int = 8192,
    context_length: int = 256,
) -> dict[str, object]:
    source = {
        **TINYSTORIES_SOURCE,
        "selection": "byte prefix ending at the last complete <|endoftext|> story",
    }
    return build_dataset_artifacts(
        train_bytes=_download_prefix("TinyStories-train.txt", train_bytes),
        validation_bytes=_download_prefix("TinyStories-valid.txt", validation_bytes),
        output_dir=output_dir,
        source=source,
        vocab_size=vocab_size,
        context_length=context_length,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen OrcaColony tokenizer and packed dataset artifacts"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--validation-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=256)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = build_tinystories_subset(
        args.output,
        train_bytes=args.train_bytes,
        validation_bytes=args.validation_bytes,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "train_sequences": manifest["packing"]["train_sequences"],
                "validation_sequences": manifest["packing"][
                    "validation_sequences"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
