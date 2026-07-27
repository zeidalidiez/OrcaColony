from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TASK_ID = "record-patch-v1"
CAMPAIGN_ID = "orcacolony-record-patch-t2-v1"
DATASET_LICENSE = "cc0-1.0"
MODEL_LICENSE = "apache-2.0"
END_OF_RECORD = "<|endoftext|>"
PUBLIC_TRAIN_KEY = (
    "19ce6db722c32c4be992706061422c1a13844d9c12e37af0528148633846daed"
)
PUBLIC_LANGUAGE_VALIDATION_KEY = (
    "3aa3d252488910aec2858b36eb0cf0c6717a655f274fdc72d972e2163bbd2c0e"
)
PUBLIC_BEHAVIORAL_VALIDATION_KEY = (
    "b410762700174b5f0569ff5d99cf6331d88cfb9025213447729e451813023824"
)
BEHAVIORAL_BUCKETS = (
    "set-existing",
    "set-new",
    "delete-existing",
    "rename-new",
    "overwrite-chain",
    "delete-readd",
    "rename-collision",
    "mixed-sequence",
)
FIELD_NAMES = (
    "active",
    "category",
    "color",
    "group",
    "mode",
    "name",
    "owner",
    "priority",
    "region",
    "status",
    "tag",
    "tier",
    "type",
    "zone",
)
STRING_PARTS = (
    "amber",
    "birch",
    "cedar",
    "delta",
    "ember",
    "fjord",
    "grove",
    "harbor",
    "indigo",
    "juniper",
    "kestrel",
    "lumen",
)
T2_PARAMETER_COUNT = 17_538_816
DEFAULT_TRAIN_EXAMPLES = 32_768
DEFAULT_LANGUAGE_VALIDATION_EXAMPLES = 1_024
DEFAULT_BEHAVIORAL_VALIDATION_EXAMPLES = 32
DEFAULT_BEHAVIORAL_FINAL_HOLDOUT_EXAMPLES = 128


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_revision() -> str:
    return _sha256_file(Path(__file__).resolve())


def _write_exact(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    path.write_bytes(payload)
    if mode is not None:
        path.chmod(mode)


def _load_mapping(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


class _HashStream:
    """Small deterministic byte stream with no dependency on random.Random."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("deterministic stream key may not be empty")
        self._key = key
        self._counter = 0

    def _word(self) -> int:
        payload = hashlib.sha256(
            self._key + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        return int.from_bytes(payload[:8], "big")

    def below(self, limit: int) -> int:
        if limit <= 0:
            raise ValueError("deterministic choice limit must be positive")
        ceiling = (1 << 64) - ((1 << 64) % limit)
        while True:
            value = self._word()
            if value < ceiling:
                return value % limit

    def choice(self, values: Sequence[object]) -> object:
        if not values:
            raise ValueError("deterministic choice requires values")
        return values[self.below(len(values))]

    def shuffled(self, values: Sequence[object]) -> list[object]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            swap = self.below(index + 1)
            result[index], result[swap] = result[swap], result[index]
        return result


def _example_stream(key_hex: str, split: str, index: int) -> _HashStream:
    if (
        len(key_hex) != 64
        or any(character not in "0123456789abcdef" for character in key_hex)
    ):
        raise ValueError("split key must be 64 lowercase hexadecimal characters")
    if index < 0:
        raise ValueError("example index must be nonnegative")
    key = hashlib.sha256(
        bytes.fromhex(key_hex)
        + b"\x00"
        + split.encode("ascii")
        + b"\x00"
        + index.to_bytes(8, "big")
    ).digest()
    return _HashStream(key)


def _value(stream: _HashStream) -> object:
    kind = stream.below(5)
    if kind == 0:
        left = str(stream.choice(STRING_PARTS))
        right = str(stream.choice(STRING_PARTS))
        suffix = stream.below(100)
        return f"{left}-{right}-{suffix:02d}"
    if kind == 1:
        return stream.below(200) - 50
    if kind == 2:
        return bool(stream.below(2))
    if kind == 3:
        return None
    return str(stream.choice(STRING_PARTS))


def _different_value(stream: _HashStream, current: object) -> object:
    current_json = _compact_json(current)
    for _ in range(32):
        candidate = _value(stream)
        if _compact_json(candidate) != current_json:
            return candidate
    raise RuntimeError("could not generate a distinct value")


def _initial_record(stream: _HashStream) -> dict[str, object]:
    count = 4 + stream.below(3)
    keys = [
        str(value)
        for value in stream.shuffled(FIELD_NAMES)[:count]
    ]
    return {key: _value(stream) for key in keys}


def _missing_key(stream: _HashStream, record: Mapping[str, object]) -> str:
    missing = [key for key in FIELD_NAMES if key not in record]
    return str(stream.choice(missing))


def _set_operation(key: str, value: object) -> dict[str, object]:
    return {"op": "set", "key": key, "value": value}


def _delete_operation(key: str) -> dict[str, object]:
    return {"op": "delete", "key": key}


def _rename_operation(source: str, target: str) -> dict[str, object]:
    return {"op": "rename", "source": source, "target": target}


def _validate_scalar(value: object, label: str) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    raise ValueError(f"{label} must be a JSON string, integer, boolean, or null")


def apply_patch(
    record: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if (
        not record
        or any(
            not isinstance(key, str) or key not in FIELD_NAMES
            for key in record
        )
    ):
        raise ValueError("record keys are invalid")
    for key, value in record.items():
        _validate_scalar(value, f"record value for {key}")
    result = dict(record)
    for operation in operations:
        op = operation.get("op")
        if op == "set" and set(operation) == {"op", "key", "value"}:
            key = operation.get("key")
            if not isinstance(key, str) or key not in FIELD_NAMES:
                raise ValueError("set operation key is invalid")
            value = operation.get("value")
            _validate_scalar(value, f"set value for {key}")
            result[key] = value
        elif op == "delete" and set(operation) == {"op", "key"}:
            key = operation.get("key")
            if not isinstance(key, str) or key not in FIELD_NAMES:
                raise ValueError("delete operation key is invalid")
            result.pop(key, None)
        elif op == "rename" and set(operation) == {
            "op",
            "source",
            "target",
        }:
            source = operation.get("source")
            target = operation.get("target")
            if (
                not isinstance(source, str)
                or source not in FIELD_NAMES
                or not isinstance(target, str)
                or target not in FIELD_NAMES
                or source == target
            ):
                raise ValueError("rename operation keys are invalid")
            if source in result:
                value = result.pop(source)
                result[target] = value
        else:
            raise ValueError("unsupported patch operation")
    return result


def _operations_for_bucket(
    stream: _HashStream,
    record: Mapping[str, object],
    bucket: str,
) -> list[dict[str, object]]:
    present = list(record)
    source = str(stream.choice(present))
    missing = _missing_key(stream, record)
    if bucket == "set-existing":
        return [_set_operation(source, _different_value(stream, record[source]))]
    if bucket == "set-new":
        return [_set_operation(missing, _value(stream))]
    if bucket == "delete-existing":
        return [_delete_operation(source)]
    if bucket == "rename-new":
        return [_rename_operation(source, missing)]
    if bucket == "overwrite-chain":
        first = _different_value(stream, record[source])
        second = _different_value(stream, first)
        return [_set_operation(source, first), _set_operation(source, second)]
    if bucket == "delete-readd":
        return [
            _delete_operation(source),
            _set_operation(source, _different_value(stream, record[source])),
        ]
    if bucket == "rename-collision":
        target = str(stream.choice([key for key in present if key != source]))
        return [_rename_operation(source, target)]
    if bucket == "mixed-sequence":
        remaining = [key for key in present if key != source]
        delete_key = str(stream.choice(remaining))
        rename_source = str(
            stream.choice(
                [key for key in remaining if key != delete_key]
            )
        )
        rename_target = _missing_key(stream, record)
        second_missing = str(
            stream.choice(
                [
                    key
                    for key in FIELD_NAMES
                    if key not in record and key != rename_target
                ]
            )
        )
        return [
            _set_operation(source, _different_value(stream, record[source])),
            _delete_operation(delete_key),
            _rename_operation(rename_source, rename_target),
            _set_operation(second_missing, _value(stream)),
        ]
    raise ValueError(f"unknown behavioral bucket: {bucket}")


def _render_operation(operation: Mapping[str, object]) -> str:
    op = operation["op"]
    if op == "set":
        return f"SET {operation['key']} {_compact_json(operation['value'])}"
    if op == "delete":
        return f"DELETE {operation['key']}"
    if op == "rename":
        return f"RENAME {operation['source']} {operation['target']}"
    raise ValueError("unsupported patch operation")


def _parse_prompt(prompt: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    lines = prompt.splitlines()
    if (
        len(lines) < 5
        or lines[0] != "record_patch_v1"
        or not lines[1].startswith("record ")
        or lines[2] != "patch"
        or lines[-1] != "result"
    ):
        raise ValueError("behavioral prompt framing is invalid")
    try:
        record = json.loads(
            lines[1].removeprefix("record "),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("behavioral prompt record is invalid") from exc
    if not isinstance(record, dict):
        raise ValueError("behavioral prompt record must be an object")
    operations: list[dict[str, object]] = []
    for line in lines[3:-1]:
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[0] == "SET":
            key = parts[1]
            try:
                value = json.loads(
                    parts[2],
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("behavioral prompt SET value is invalid") from exc
            operations.append(_set_operation(key, value))
        elif len(parts) == 2 and parts[0] == "DELETE":
            operations.append(_delete_operation(parts[1]))
        elif len(parts) == 3 and parts[0] == "RENAME":
            operations.append(_rename_operation(parts[1], parts[2]))
        else:
            raise ValueError("behavioral prompt operation is invalid")
    if not operations:
        raise ValueError("behavioral prompt has no operations")
    apply_patch(record, operations)
    return record, operations


def generate_example(
    *,
    key_hex: str,
    split: str,
    index: int,
) -> dict[str, object]:
    stream = _example_stream(key_hex, split, index)
    bucket = BEHAVIORAL_BUCKETS[index % len(BEHAVIORAL_BUCKETS)]
    record = _initial_record(stream)
    operations = _operations_for_bucket(stream, record, bucket)
    expected = apply_patch(record, operations)
    shuffled_record = {
        str(key): record[str(key)]
        for key in stream.shuffled(list(record))
    }
    record_text = json.dumps(
        shuffled_record,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    operation_text = "\n".join(
        _render_operation(operation) for operation in operations
    )
    prompt = (
        "record_patch_v1\n"
        f"record {record_text}\n"
        "patch\n"
        f"{operation_text}\n"
        "result\n"
    )
    target = _compact_json(expected)
    prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
    example_id = (
        f"{TASK_ID}-{split}-{index:06d}-{prompt_sha256[:12]}"
    )
    return {
        "id": example_id,
        "split": split,
        "bucket": bucket,
        "record": record,
        "operations": operations,
        "prompt": prompt,
        "target": target,
    }


def _examples(
    *,
    key_hex: str,
    split: str,
    count: int,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("split example count must be positive")
    if count % len(BEHAVIORAL_BUCKETS):
        raise ValueError(
            "split example count must be divisible by the behavioral bucket count"
        )
    examples = [
        generate_example(key_hex=key_hex, split=split, index=index)
        for index in range(count)
    ]
    prompts = [str(example["prompt"]) for example in examples]
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"{split} contains duplicate prompts")
    return examples


def _jsonl(examples: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            dict(example),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for example in examples
    ).encode("utf-8")


def _transcript_corpus(examples: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        f"{example['prompt']}{example['target']}\n{END_OF_RECORD}\n"
        for example in examples
    ).encode("utf-8")


def _load_or_create_holdout_key(path: Path) -> str:
    if path.exists():
        payload = _load_mapping(path, "holdout key")
        if set(payload) != {"format", "task_id", "key"}:
            raise ValueError("holdout key schema is invalid")
        if (
            payload.get("format") != "orcacolony_holdout_key_v1"
            or payload.get("task_id") != TASK_ID
        ):
            raise ValueError("holdout key identity is invalid")
        key = payload.get("key")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise ValueError("holdout key is invalid")
        return key
    key = secrets.token_hex(32)
    payload = {
        "format": "orcacolony_holdout_key_v1",
        "task_id": TASK_ID,
        "key": key,
    }
    _write_exact(path, _canonical_json_bytes(payload), mode=0o600)
    return key


def _public_checksums(public_dir: Path, names: Sequence[str]) -> None:
    records = "".join(
        f"{_sha256_file(public_dir / name)}  {name}\n"
        for name in sorted(names)
    ).encode("utf-8")
    _write_exact(public_dir / "SHA256SUMS", records)


@dataclass(frozen=True)
class FrozenRecordPatch:
    public_dir: Path
    private_dir: Path
    packed_dir: Path
    campaign_path: Path
    behavioral_suite_revision: str
    evaluator_revision: str
    initialization_revision: str


def _build_campaign_payload(
    *,
    packed_manifest: Mapping[str, object],
    behavioral_suite_revision: str,
    evaluator_revision: str,
    initialization_revision: str,
    steps: int,
) -> dict[str, object]:
    packing = packed_manifest.get("packing")
    files = packed_manifest.get("files")
    tokenizer = packed_manifest.get("tokenizer")
    if (
        not isinstance(packing, Mapping)
        or not isinstance(files, Mapping)
        or not isinstance(tokenizer, Mapping)
    ):
        raise ValueError("packed dataset manifest is incomplete")
    train_sequences = packing.get("train_sequences")
    validation_sequences = packing.get("validation_sequences")
    if (
        isinstance(train_sequences, bool)
        or not isinstance(train_sequences, int)
        or train_sequences <= 0
        or isinstance(validation_sequences, bool)
        or not isinstance(validation_sequences, int)
        or validation_sequences < 2
    ):
        raise ValueError("packed dataset sequence counts are invalid")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("campaign steps must be positive")
    validation_count = validation_sequences // 2
    holdout_count = validation_sequences - validation_count
    validation_batch_size = min(4, validation_count)
    holdout_batch_size = min(4, holdout_count)
    return {
        "campaign": {
            "id": CAMPAIGN_ID,
            "objective": "causal_lm",
            "loss_mask": "all_target_tokens",
        },
        "model": {
            "architecture": "volunteer_decoder_v1",
            "architecture_revision": 1,
            "layers": 8,
            "width": 384,
            "heads": 6,
            "mlp_width": 1536,
            "vocabulary_size": 8192,
            "context_length": 512,
            "positional_encoding": "learned_absolute",
            "layer_norm_epsilon": 0.00001,
            "gelu_approximation": "tanh",
            "attention_bias": True,
            "linear_bias": True,
            "tied_token_embeddings": True,
            "parameters": T2_PARAMETER_COUNT,
        },
        "training": {
            "seed": 20260726,
            "batch_size": 4,
            "dataset_sequences": train_sequences,
            "active_vocabulary_size": 8192,
            "steps": steps,
            "learning_rate": 0.0003,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_gradient_norm": 1.0,
            "compute_dtype": "float32",
            "gradient_accumulation_dtype": "float32",
        },
        "dataset": {
            "format": "orcacolony_dataset_artifacts_v1",
            "manifest_sha256": _sha256_bytes(
                _canonical_json_bytes(dict(packed_manifest))
            ),
            "tokenizer_sha256": tokenizer.get("sha256"),
            "train_sha256": files.get("train.safetensors"),
            "validation_sha256": files.get("validation.safetensors"),
        },
        "evaluation": {
            "metric": "held_out_cross_entropy",
            "checkpoint_selection": "lowest_mean_loss",
            "validation_start_sequence": 0,
            "validation_sequences": validation_count,
            "batch_size": validation_batch_size,
            "final_holdout": {
                "start_sequence": validation_count,
                "sequence_count": holdout_count,
                "batch_size": holdout_batch_size,
            },
            "success_gate": {
                "metric": "mean_loss",
                "minimum_improvement_from_initialization": 0.1,
            },
        },
        "research": {
            "format": "orcacolony_capability_research_v1",
            "claim": (
                "Training improves exact execution of ordered flat-record "
                "SET, DELETE, and RENAME patches on the frozen Record Patch v1 "
                "suite."
            ),
            "baseline": {
                "id": "t2-random-initialization",
                "description": (
                    "The exact deterministic 17,538,816-parameter T2 "
                    "initialization evaluated with greedy decoding."
                ),
                "revision": f"sha256:{initialization_revision}",
            },
            "primary_metric": {
                "id": "record-exact-match",
                "description": (
                    "Fraction of examples whose stripped output is exactly the "
                    "required canonical JSON record."
                ),
                "direction": "maximize",
                "unit": "ratio",
                "success_threshold": 0.7,
                "minimum_improvement_from_baseline": 0.2,
            },
            "guardrails": [
                {
                    "id": "valid-json-rate",
                    "description": (
                        "At least 95% of outputs must be one valid JSON value "
                        "with no duplicate object keys."
                    ),
                },
                {
                    "id": "canonical-json-rate",
                    "description": (
                        "At least 90% of outputs must use the exact canonical "
                        "one-line JSON serialization."
                    ),
                },
                {
                    "id": "single-operation-exact-match",
                    "description": (
                        "Exact match across the four single-operation buckets "
                        "must be at least 80%."
                    ),
                },
            ],
            "analysis_plan": [
                (
                    "Publish every sample ID, prompt, expected output, model "
                    "output, parse status, semantic status, and exact-match "
                    "status."
                ),
                (
                    "Compare initialization and checkpoints by operation "
                    "bucket, output validity, canonical formatting, and error "
                    "type."
                ),
                (
                    "Inspect language-loss, gradient norms, clipping, update "
                    "norms, duplicate prompts, nearest training records, and "
                    "memorization indicators before attributing a cause."
                ),
                (
                    "Treat the all-target causal transcript objective as a "
                    "measured limitation, not as target-only supervised fine-"
                    "tuning."
                ),
            ],
            "final_holdout_policy": (
                "release_only_after_checkpoint_selection"
            ),
            "checkpoint_selection": (
                "lowest_validation_mean_loss_before_behavioral_final_holdout"
            ),
            "behavioral_evaluation": {
                "suite_id": TASK_ID,
                "dataset_revision": (
                    f"sha256:{behavioral_suite_revision}"
                ),
                "evaluator_revision": f"sha256:{evaluator_revision}",
                "validation_split": "behavioral_validation",
                "final_holdout_split": "behavioral_final_holdout",
            },
        },
        "publication": {
            "format": "orcacolony_huggingface_publication_v1",
            "model_repo_id": "OrcaColony/record-patch-t2-v1",
            "dataset_repo_id": "OrcaColony/record-patch-v1",
            "model_license": MODEL_LICENSE,
            "dataset_license": DATASET_LICENSE,
            "visibility_policy": "private_review_then_public",
        },
    }


def freeze_record_patch(
    *,
    public_dir: str | Path,
    private_dir: str | Path,
    campaign_path: str | Path,
    train_examples: int = DEFAULT_TRAIN_EXAMPLES,
    language_validation_examples: int = (
        DEFAULT_LANGUAGE_VALIDATION_EXAMPLES
    ),
    behavioral_validation_examples: int = (
        DEFAULT_BEHAVIORAL_VALIDATION_EXAMPLES
    ),
    behavioral_final_holdout_examples: int = (
        DEFAULT_BEHAVIORAL_FINAL_HOLDOUT_EXAMPLES
    ),
    steps: int = 2_048,
) -> FrozenRecordPatch:
    from .artifacts import PackedDataset, build_dataset_artifacts
    from .reference import (
        build_model,
        campaign_from_mapping,
        configure_determinism,
        tensor_sha256,
        validate_dataset_artifacts,
    )

    public_root = Path(public_dir).resolve()
    private_root = Path(private_dir).resolve()
    campaign_file = Path(campaign_path).resolve()
    public_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    module_revision = _module_revision()

    validation_examples = _examples(
        key_hex=PUBLIC_BEHAVIORAL_VALIDATION_KEY,
        split="behavioral_validation",
        count=behavioral_validation_examples,
    )
    holdout_key = _load_or_create_holdout_key(
        private_root / "holdout-key.json"
    )
    final_holdout_examples = _examples(
        key_hex=holdout_key,
        split="behavioral_final_holdout",
        count=behavioral_final_holdout_examples,
    )
    validation_prompts = {
        str(example["prompt"]) for example in validation_examples
    }
    final_prompts = {
        str(example["prompt"]) for example in final_holdout_examples
    }
    if validation_prompts.intersection(final_prompts):
        raise ValueError("behavioral validation and final holdout overlap")

    validation_bytes = _jsonl(validation_examples)
    final_holdout_bytes = _jsonl(final_holdout_examples)
    _write_exact(
        public_root / "behavioral-validation.jsonl",
        validation_bytes,
    )
    _write_exact(
        private_root / "behavioral-final-holdout.jsonl",
        final_holdout_bytes,
        mode=0o600,
    )
    recipe = {
        "format": "orcacolony_record_patch_recipe_v1",
        "task_id": TASK_ID,
        "license": DATASET_LICENSE,
        "generator_revision": f"sha256:{module_revision}",
        "objective": {
            "name": "causal_lm",
            "loss_mask": "all_target_tokens",
            "transcript": (
                "prompt followed by canonical target, newline, and "
                f"{END_OF_RECORD}"
            ),
            "limitation": (
                "Prompt and answer tokens both carry loss. This is not "
                "target-only supervised fine-tuning."
            ),
        },
        "splits": {
            "train": {
                "examples": train_examples,
                "key": PUBLIC_TRAIN_KEY,
            },
            "language_validation": {
                "examples": language_validation_examples,
                "key": PUBLIC_LANGUAGE_VALIDATION_KEY,
            },
            "behavioral_validation": {
                "examples": behavioral_validation_examples,
                "key": PUBLIC_BEHAVIORAL_VALIDATION_KEY,
            },
            "behavioral_final_holdout": {
                "examples": behavioral_final_holdout_examples,
                "key": "withheld_until_checkpoint_selection",
            },
        },
        "behavioral_buckets": list(BEHAVIORAL_BUCKETS),
    }
    recipe_bytes = _canonical_json_bytes(recipe)
    _write_exact(public_root / "training-recipe.json", recipe_bytes)

    suite_lock = {
        "format": "orcacolony_behavioral_suite_v1",
        "task_id": TASK_ID,
        "license": DATASET_LICENSE,
        "generator_revision": f"sha256:{module_revision}",
        "splits": {
            "behavioral_validation": {
                "examples": behavioral_validation_examples,
                "file": "behavioral-validation.jsonl",
                "sha256": _sha256_bytes(validation_bytes),
                "public_before_training": True,
            },
            "behavioral_final_holdout": {
                "examples": behavioral_final_holdout_examples,
                "file": "behavioral-final-holdout.jsonl",
                "sha256": _sha256_bytes(final_holdout_bytes),
                "public_before_training": False,
            },
        },
        "separation": {
            "prompt_overlap": 0,
            "final_holdout_key": "withheld_until_checkpoint_selection",
        },
    }
    suite_lock_bytes = _canonical_json_bytes(suite_lock)
    behavioral_suite_revision = _sha256_bytes(suite_lock_bytes)
    _write_exact(
        public_root / "behavioral-suite-lock.json",
        suite_lock_bytes,
    )
    _public_checksums(
        public_root,
        (
            "behavioral-suite-lock.json",
            "behavioral-validation.jsonl",
            "training-recipe.json",
        ),
    )

    train = _examples(
        key_hex=PUBLIC_TRAIN_KEY,
        split="train",
        count=train_examples,
    )
    language_validation = _examples(
        key_hex=PUBLIC_LANGUAGE_VALIDATION_KEY,
        split="language_validation",
        count=language_validation_examples,
    )
    all_non_holdout_prompts = {
        *(str(example["prompt"]) for example in train),
        *(str(example["prompt"]) for example in language_validation),
        *validation_prompts,
    }
    if len(all_non_holdout_prompts) != (
        len(train) + len(language_validation) + len(validation_examples)
    ):
        raise ValueError("non-holdout task splits overlap")
    if all_non_holdout_prompts.intersection(final_prompts):
        raise ValueError("final holdout overlaps another task split")

    train_bytes = _transcript_corpus(train)
    language_validation_bytes = _transcript_corpus(language_validation)
    source_dir = private_root / "source"
    _write_exact(source_dir / "train.txt", train_bytes)
    _write_exact(
        source_dir / "language-validation.txt",
        language_validation_bytes,
    )
    source_lock = {
        "format": "orcacolony_record_patch_source_lock_v1",
        "task_id": TASK_ID,
        "license": DATASET_LICENSE,
        "recipe_sha256": _sha256_bytes(recipe_bytes),
        "behavioral_suite_revision": (
            f"sha256:{behavioral_suite_revision}"
        ),
        "files": {
            "train.txt": _sha256_bytes(train_bytes),
            "language-validation.txt": _sha256_bytes(
                language_validation_bytes
            ),
        },
    }
    source_lock_bytes = _canonical_json_bytes(source_lock)
    source_revision = _sha256_bytes(source_lock_bytes)
    _write_exact(source_dir / "source-lock.json", source_lock_bytes)

    packed_dir = private_root / "packed"
    if packed_dir.exists() and any(packed_dir.iterdir()):
        dataset = PackedDataset.load(packed_dir)
        packed_manifest = dict(dataset.manifest)
    else:
        packed_manifest = build_dataset_artifacts(
            train_bytes=train_bytes,
            validation_bytes=language_validation_bytes,
            output_dir=packed_dir,
            source={
                "dataset": "OrcaColony/record-patch-v1",
                "revision": f"sha256:{source_revision}",
                "license": DATASET_LICENSE,
                "license_url": (
                    "https://creativecommons.org/publicdomain/zero/1.0/"
                ),
                "selection": (
                    "Deterministic project-generated Record Patch v1 "
                    "transcripts with separately keyed splits."
                ),
            },
            vocab_size=8192,
            context_length=512,
            notice_changes=(
                "OrcaColony generated deterministic synthetic flat-record "
                "patch examples, kept training and language-validation keys "
                "separate, trained a byte-level BPE tokenizer on training "
                "transcripts only, and packed shifted causal-LM tensors. No "
                "external text or teacher-model output is present."
            ),
        )
        dataset = PackedDataset.load(packed_dir)

    configure_determinism(20260726)
    model_only_payload = {
        "campaign": {
            "id": CAMPAIGN_ID,
            "objective": "causal_lm",
            "loss_mask": "all_target_tokens",
        },
        "model": {
            "architecture": "volunteer_decoder_v1",
            "architecture_revision": 1,
            "layers": 8,
            "width": 384,
            "heads": 6,
            "mlp_width": 1536,
            "vocabulary_size": 8192,
            "context_length": 512,
            "positional_encoding": "learned_absolute",
            "layer_norm_epsilon": 0.00001,
            "gelu_approximation": "tanh",
            "attention_bias": True,
            "linear_bias": True,
            "tied_token_embeddings": True,
            "parameters": T2_PARAMETER_COUNT,
        },
        "training": {
            "seed": 20260726,
            "batch_size": 4,
            "dataset_sequences": int(
                packed_manifest["packing"]["train_sequences"]  # type: ignore[index]
            ),
            "active_vocabulary_size": 8192,
            "steps": steps,
            "learning_rate": 0.0003,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_gradient_norm": 1.0,
            "compute_dtype": "float32",
            "gradient_accumulation_dtype": "float32",
        },
    }
    model_config = campaign_from_mapping(model_only_payload)
    model = build_model(model_config)
    if sum(parameter.numel() for parameter in model.parameters()) != (
        T2_PARAMETER_COUNT
    ):
        raise ValueError("true T2 parameter count differs")
    initialization_revision = tensor_sha256(model.state_dict())
    campaign_payload = _build_campaign_payload(
        packed_manifest=packed_manifest,
        behavioral_suite_revision=behavioral_suite_revision,
        evaluator_revision=module_revision,
        initialization_revision=initialization_revision,
        steps=steps,
    )
    campaign = campaign_from_mapping(campaign_payload)
    validate_dataset_artifacts(campaign, dataset)
    campaign_bytes = _canonical_json_bytes(campaign_payload)
    _write_exact(campaign_file, campaign_bytes)
    bundle = {
        "format": "orcacolony_record_patch_freeze_v1",
        "task_id": TASK_ID,
        "campaign_id": CAMPAIGN_ID,
        "campaign_file": str(campaign_file),
        "campaign_sha256": _sha256_bytes(campaign_bytes),
        "behavioral_suite_revision": (
            f"sha256:{behavioral_suite_revision}"
        ),
        "evaluator_revision": f"sha256:{module_revision}",
        "initialization_revision": f"sha256:{initialization_revision}",
        "source_revision": f"sha256:{source_revision}",
        "packed_dataset_revision": f"sha256:{dataset.revision}",
        "holdout": {
            "state": "withheld",
            "file": "behavioral-final-holdout.jsonl",
            "sha256": _sha256_bytes(final_holdout_bytes),
        },
    }
    _write_exact(
        private_root / "freeze-record.json",
        _canonical_json_bytes(bundle),
        mode=0o600,
    )
    return FrozenRecordPatch(
        public_dir=public_root,
        private_dir=private_root,
        packed_dir=packed_dir,
        campaign_path=campaign_file,
        behavioral_suite_revision=behavioral_suite_revision,
        evaluator_revision=module_revision,
        initialization_revision=initialization_revision,
    )


def _parse_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            raise ValueError(f"{label} contains a blank line")
        try:
            row = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def load_behavioral_split(
    *,
    public_dir: str | Path,
    split: str,
    final_holdout_path: str | Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public_root = Path(public_dir).resolve()
    lock_path = public_root / "behavioral-suite-lock.json"
    lock = _load_mapping(lock_path, "behavioral suite lock")
    if (
        lock.get("format") != "orcacolony_behavioral_suite_v1"
        or lock.get("task_id") != TASK_ID
    ):
        raise ValueError("behavioral suite identity is invalid")
    splits = lock.get("splits")
    if not isinstance(splits, Mapping) or split not in splits:
        raise ValueError("behavioral split is not declared")
    split_lock = splits[split]
    if not isinstance(split_lock, Mapping):
        raise ValueError("behavioral split lock is invalid")
    if split == "behavioral_validation":
        path = public_root / "behavioral-validation.jsonl"
    elif split == "behavioral_final_holdout":
        if final_holdout_path is None:
            raise ValueError(
                "final holdout path is required after checkpoint selection"
            )
        path = Path(final_holdout_path).resolve()
    else:
        raise ValueError("behavioral split is unsupported")
    if not path.is_file() or _sha256_file(path) != split_lock.get("sha256"):
        raise ValueError("behavioral split digest differs from its lock")
    rows = _parse_jsonl(path, "behavioral split")
    if len(rows) != split_lock.get("examples"):
        raise ValueError("behavioral split count differs from its lock")
    identifiers: set[str] = set()
    for row in rows:
        if set(row) != {
            "id",
            "split",
            "bucket",
            "record",
            "operations",
            "prompt",
            "target",
        }:
            raise ValueError("behavioral example schema is invalid")
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or identifier in identifiers
            or row.get("split") != split
            or row.get("bucket") not in BEHAVIORAL_BUCKETS
            or not isinstance(row.get("record"), Mapping)
            or not isinstance(row.get("operations"), list)
            or not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("target"), str)
        ):
            raise ValueError("behavioral example identity is invalid")
        identifiers.add(identifier)
        prompt = str(row["prompt"])
        if (
            identifier.rsplit("-", 1)[-1]
            != _sha256_bytes(prompt.encode("utf-8"))[:12]
        ):
            raise ValueError("behavioral example ID does not bind its prompt")
        prompt_record, prompt_operations = _parse_prompt(prompt)
        if (
            _compact_json(prompt_record)
            != _compact_json(row["record"])
            or prompt_operations != row["operations"]
        ):
            raise ValueError("behavioral prompt differs from its oracle fields")
        expected = apply_patch(
            row["record"],  # type: ignore[arg-type]
            row["operations"],  # type: ignore[arg-type]
        )
        if row["target"] != _compact_json(expected):
            raise ValueError("behavioral example target differs from oracle")
    return lock, rows


def _strict_json_output(output: str) -> tuple[bool, object | None, str | None]:
    stripped = output.strip()
    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError):
        return False, None, None
    return True, parsed, _compact_json(parsed)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires observations")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_predictions(
    *,
    public_dir: str | Path,
    split: str,
    predictions_path: str | Path,
    final_holdout_path: str | Path | None = None,
) -> dict[str, object]:
    lock, examples = load_behavioral_split(
        public_dir=public_dir,
        split=split,
        final_holdout_path=final_holdout_path,
    )
    prediction_rows = _parse_jsonl(
        Path(predictions_path).resolve(),
        "prediction file",
    )
    predictions: dict[str, str] = {}
    for row in prediction_rows:
        if set(row) != {"id", "output"}:
            raise ValueError("prediction row schema is invalid")
        identifier = row.get("id")
        output = row.get("output")
        if (
            not isinstance(identifier, str)
            or identifier in predictions
            or not isinstance(output, str)
        ):
            raise ValueError("prediction row is invalid")
        predictions[identifier] = output
    expected_ids = {str(example["id"]) for example in examples}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(
            "prediction IDs differ from suite"
            f"; missing={missing[:3]}; extra={extra[:3]}"
        )

    samples: list[dict[str, object]] = []
    exact_count = 0
    semantic_count = 0
    valid_count = 0
    canonical_count = 0
    bucket_totals = {
        bucket: {
            "examples": 0,
            "exact_matches": 0,
            "semantic_matches": 0,
            "valid_json": 0,
            "canonical_json": 0,
        }
        for bucket in BEHAVIORAL_BUCKETS
    }
    for example in examples:
        identifier = str(example["id"])
        expected = str(example["target"])
        output = predictions[identifier]
        stripped = output.strip()
        valid, parsed, canonical = _strict_json_output(output)
        semantic = valid and canonical == expected
        canonical_output = valid and canonical == stripped
        exact = stripped == expected
        bucket = str(example["bucket"])
        exact_count += int(exact)
        semantic_count += int(semantic)
        valid_count += int(valid)
        canonical_count += int(canonical_output)
        bucket_result = bucket_totals[bucket]
        bucket_result["examples"] += 1
        bucket_result["exact_matches"] += int(exact)
        bucket_result["semantic_matches"] += int(semantic)
        bucket_result["valid_json"] += int(valid)
        bucket_result["canonical_json"] += int(canonical_output)
        error = "none"
        if not valid:
            error = "invalid-json"
        elif not semantic:
            error = "wrong-record"
        elif not canonical_output:
            error = "noncanonical-json"
        samples.append(
            {
                "id": identifier,
                "bucket": bucket,
                "prompt": example["prompt"],
                "expected": expected,
                "output": output,
                "valid_json": valid,
                "semantic_match": semantic,
                "canonical_json": canonical_output,
                "exact_match": exact,
                "error": error,
                "parsed_output": parsed,
            }
        )
    total = len(examples)
    single_buckets = {
        "set-existing",
        "set-new",
        "delete-existing",
        "rename-new",
    }
    single_total = sum(
        int(bucket_totals[bucket]["examples"]) for bucket in single_buckets
    )
    single_exact = sum(
        int(bucket_totals[bucket]["exact_matches"])
        for bucket in single_buckets
    )
    exact_low, exact_high = _wilson_interval(exact_count, total)
    result = {
        "format": "orcacolony_record_patch_evaluation_v1",
        "task_id": TASK_ID,
        "split": split,
        "behavioral_suite_revision": (
            f"sha256:{_sha256_bytes(_canonical_json_bytes(lock))}"
        ),
        "evaluator_revision": f"sha256:{_module_revision()}",
        "predictions_sha256": _sha256_file(
            Path(predictions_path).resolve()
        ),
        "metrics": {
            "examples": total,
            "record_exact_match": exact_count / total,
            "record_exact_match_count": exact_count,
            "record_exact_match_wilson_95": {
                "low": exact_low,
                "high": exact_high,
            },
            "semantic_match": semantic_count / total,
            "valid_json": valid_count / total,
            "canonical_json": canonical_count / total,
            "single_operation_exact_match": (
                single_exact / single_total
            ),
        },
        "buckets": [
            {
                "id": bucket,
                **bucket_totals[bucket],
                "exact_match": (
                    int(bucket_totals[bucket]["exact_matches"])
                    / int(bucket_totals[bucket]["examples"])
                ),
            }
            for bucket in BEHAVIORAL_BUCKETS
        ],
        "guardrails": [
            {
                "id": "valid-json-rate",
                "threshold": 0.95,
                "value": valid_count / total,
                "passed": valid_count / total >= 0.95,
            },
            {
                "id": "canonical-json-rate",
                "threshold": 0.9,
                "value": canonical_count / total,
                "passed": canonical_count / total >= 0.9,
            },
            {
                "id": "single-operation-exact-match",
                "threshold": 0.8,
                "value": single_exact / single_total,
                "passed": single_exact / single_total >= 0.8,
            },
        ],
        "samples": samples,
    }
    return result


def _prediction_bytes(predictions: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            dict(prediction),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for prediction in predictions
    ).encode("utf-8")


def write_oracle_predictions(
    *,
    public_dir: str | Path,
    split: str,
    output_path: str | Path,
    final_holdout_path: str | Path | None = None,
) -> None:
    _, examples = load_behavioral_split(
        public_dir=public_dir,
        split=split,
        final_holdout_path=final_holdout_path,
    )
    _write_exact(
        Path(output_path).resolve(),
        _prediction_bytes(
            {
                "id": example["id"],
                "output": example["target"],
            }
            for example in examples
        ),
    )


def _greedy_completion(
    model: object,
    tokenizer: object,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    import torch

    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("baseline tokenizer lacks BOS or EOS")
    prompt_ids = [bos_id, *encoded.ids]
    context_length = int(model.config.context_length)
    if len(prompt_ids) + max_new_tokens > context_length:
        raise ValueError("baseline prompt and generation budget exceed context")
    token_ids = list(prompt_ids)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(
                torch.tensor([token_ids], dtype=torch.long)
            )[0, -1]
            next_token = int(torch.argmax(logits).item())
            if next_token == eos_id:
                break
            token_ids.append(next_token)
            generated.append(next_token)
            decoded = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            if "\n" in decoded:
                return decoded.split("\n", 1)[0]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_initialization_baseline(
    *,
    campaign_path: str | Path,
    packed_dir: str | Path,
    public_dir: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 64,
) -> dict[str, object]:
    import torch
    from tokenizers import Tokenizer

    from .artifacts import PackedDataset
    from .reference import (
        build_model,
        configure_determinism,
        load_campaign,
        tensor_sha256,
        validate_dataset_artifacts,
    )

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("baseline generation budget must be positive")
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("baseline output directory is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_file = Path(campaign_path).resolve()
    campaign = load_campaign(campaign_file)
    if campaign.campaign.get("id") != CAMPAIGN_ID:
        raise ValueError("baseline campaign identity is invalid")
    dataset = PackedDataset.load(packed_dir)
    validate_dataset_artifacts(campaign, dataset)
    configure_determinism(campaign.training.seed)
    model = build_model(campaign)
    model.eval()
    model_revision = tensor_sha256(model.state_dict())
    baseline_contract = (
        campaign.research.get("baseline")
        if campaign.research is not None
        else None
    )
    if (
        not isinstance(baseline_contract, Mapping)
        or baseline_contract.get("revision")
        != f"sha256:{model_revision}"
    ):
        raise ValueError("campaign baseline revision differs from initialization")
    tokenizer = Tokenizer.from_file(
        str(Path(packed_dir).resolve() / "tokenizer.json")
    )
    _, examples = load_behavioral_split(
        public_dir=public_dir,
        split="behavioral_validation",
    )
    predictions = []
    for example in examples:
        predictions.append(
            {
                "id": example["id"],
                "output": _greedy_completion(
                    model,
                    tokenizer,
                    str(example["prompt"]),
                    max_new_tokens=max_new_tokens,
                ),
            }
        )
    predictions_path = output_root / "predictions.jsonl"
    _write_exact(predictions_path, _prediction_bytes(predictions))
    evaluation = evaluate_predictions(
        public_dir=public_dir,
        split="behavioral_validation",
        predictions_path=predictions_path,
    )
    _write_exact(
        output_root / "evaluation.json",
        _canonical_json_bytes(evaluation),
    )
    baseline = {
        "format": "orcacolony_record_patch_baseline_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": _sha256_file(campaign_file),
        "model_revision": f"sha256:{model_revision}",
        "split": "behavioral_validation",
        "decoding": {
            "method": "greedy",
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": "first newline or EOS",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "evaluation_sha256": _sha256_bytes(
            _canonical_json_bytes(evaluation)
        ),
        "metrics": evaluation["metrics"],
        "guardrails": evaluation["guardrails"],
        "limitations": [
            (
                "This is the public behavioral-validation split, not the "
                "withheld final holdout."
            ),
            (
                "The baseline is one deterministic random initialization and "
                "does not estimate variation across initialization seeds."
            ),
            (
                "No training, volunteer work, checkpoint selection, or "
                "capability promotion occurred."
            ),
        ],
    }
    _write_exact(
        output_root / "baseline.json",
        _canonical_json_bytes(baseline),
    )
    metrics = evaluation["metrics"]
    markdown = (
        "# Record Patch T2 initialization baseline\n\n"
        "This record measures the frozen random initialization before any "
        "training or volunteer contribution.\n\n"
        f"- Campaign: `{CAMPAIGN_ID}`\n"
        f"- Model revision: `sha256:{model_revision}`\n"
        "- Split: `behavioral_validation`\n"
        f"- Examples: `{metrics['examples']}`\n"
        f"- Exact match: `{metrics['record_exact_match']}`\n"
        f"- Valid JSON: `{metrics['valid_json']}`\n"
        f"- Canonical JSON: `{metrics['canonical_json']}`\n"
        f"- Single-operation exact match: "
        f"`{metrics['single_operation_exact_match']}`\n\n"
        "The withheld final holdout was not opened. This baseline does not "
        "show that the task is learnable or that training will help.\n"
    )
    _write_exact(
        output_root / "BASELINE.md",
        markdown.encode("utf-8"),
    )
    names = (
        "BASELINE.md",
        "baseline.json",
        "evaluation.json",
        "predictions.jsonl",
    )
    checksums = "".join(
        f"{_sha256_file(output_root / name)}  {name}\n"
        for name in sorted(names)
    ).encode("utf-8")
    _write_exact(output_root / "SHA256SUMS", checksums)
    return baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and evaluate the OrcaColony Record Patch task"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--public-dir", type=Path, required=True)
    freeze.add_argument("--private-dir", type=Path, required=True)
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument(
        "--train-examples",
        type=int,
        default=DEFAULT_TRAIN_EXAMPLES,
    )
    freeze.add_argument(
        "--language-validation-examples",
        type=int,
        default=DEFAULT_LANGUAGE_VALIDATION_EXAMPLES,
    )
    freeze.add_argument(
        "--behavioral-validation-examples",
        type=int,
        default=DEFAULT_BEHAVIORAL_VALIDATION_EXAMPLES,
    )
    freeze.add_argument(
        "--behavioral-final-holdout-examples",
        type=int,
        default=DEFAULT_BEHAVIORAL_FINAL_HOLDOUT_EXAMPLES,
    )
    freeze.add_argument("--steps", type=int, default=2_048)

    oracle = subparsers.add_parser("oracle-predictions")
    oracle.add_argument("--public-dir", type=Path, required=True)
    oracle.add_argument(
        "--split",
        choices=("behavioral_validation", "behavioral_final_holdout"),
        required=True,
    )
    oracle.add_argument("--final-holdout", type=Path)
    oracle.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--public-dir", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("behavioral_validation", "behavioral_final_holdout"),
        required=True,
    )
    evaluate.add_argument("--final-holdout", type=Path)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--campaign", type=Path, required=True)
    baseline.add_argument("--packed-dir", type=Path, required=True)
    baseline.add_argument("--public-dir", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "freeze":
        frozen = freeze_record_patch(
            public_dir=args.public_dir,
            private_dir=args.private_dir,
            campaign_path=args.campaign,
            train_examples=args.train_examples,
            language_validation_examples=(
                args.language_validation_examples
            ),
            behavioral_validation_examples=(
                args.behavioral_validation_examples
            ),
            behavioral_final_holdout_examples=(
                args.behavioral_final_holdout_examples
            ),
            steps=args.steps,
        )
        summary = {
            "campaign": str(frozen.campaign_path),
            "packed_dir": str(frozen.packed_dir),
            "behavioral_suite_revision": (
                f"sha256:{frozen.behavioral_suite_revision}"
            ),
            "evaluator_revision": (
                f"sha256:{frozen.evaluator_revision}"
            ),
            "initialization_revision": (
                f"sha256:{frozen.initialization_revision}"
            ),
            "final_holdout": "withheld",
        }
    elif args.command == "oracle-predictions":
        write_oracle_predictions(
            public_dir=args.public_dir,
            split=args.split,
            output_path=args.output,
            final_holdout_path=args.final_holdout,
        )
        summary = {
            "output": str(args.output),
            "split": args.split,
        }
    elif args.command == "evaluate":
        result = evaluate_predictions(
            public_dir=args.public_dir,
            split=args.split,
            predictions_path=args.predictions,
            final_holdout_path=args.final_holdout,
        )
        _write_exact(args.output.resolve(), _canonical_json_bytes(result))
        summary = {
            "output": str(args.output),
            "metrics": result["metrics"],
        }
    else:
        baseline = run_initialization_baseline(
            campaign_path=args.campaign,
            packed_dir=args.packed_dir,
            public_dir=args.public_dir,
            output_dir=args.output,
            max_new_tokens=args.max_new_tokens,
        )
        summary = {
            "output": str(args.output),
            "metrics": baseline["metrics"],
            "final_holdout": "not opened",
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
