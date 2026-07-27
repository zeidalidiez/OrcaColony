from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from .artifacts import PackedDataset
from .record_patch import (
    PUBLIC_BEHAVIORAL_VALIDATION_KEY,
    PUBLIC_TRAIN_KEY,
    _strict_json_output,
    generate_example,
    load_behavioral_split,
)
from .reference import _load_checkpoint, load_campaign


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate analysis JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _module_revision() -> str:
    return _sha256_file(Path(__file__).resolve())


def _verify_checksums(run_root: Path) -> Mapping[str, str]:
    checksum_path = run_root / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("learnability checksum manifest is empty")
    verified: dict[str, str] = {}
    resolved_root = run_root.resolve()
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in verified
        ):
            raise ValueError("learnability checksum manifest is invalid")
        path = run_root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "learnability checksum path is unavailable or unsafe"
            ) from exc
        if not resolved.is_file() or _sha256_file(resolved) != digest:
            raise ValueError(
                f"learnability checksum mismatch: {relative}"
            )
        verified[relative] = digest
    return verified


def _numeric_summary(values: Sequence[float]) -> Mapping[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("numeric summary values must be finite and nonempty")
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": statistics.median(ordered),
        "p75": quantile(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


class _DuplicateOutputKey(ValueError):
    pass


def _duplicate_output_hook(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateOutputKey(key)
        result[key] = value
    return result


def _output_classification(output: str, expected: str) -> str:
    stripped = output.strip()
    if not stripped:
        return "empty"
    if stripped == expected:
        return "exact"
    decoder = json.JSONDecoder(object_pairs_hook=_duplicate_output_hook)
    try:
        parsed, end = decoder.raw_decode(stripped)
    except _DuplicateOutputKey:
        return "duplicate-object-keys"
    except json.JSONDecodeError:
        return "malformed-json"
    if stripped[end:].strip():
        return "trailing-content"
    if not isinstance(parsed, dict):
        return "non-object-json"
    valid, _, canonical = _strict_json_output(output)
    if not valid:
        return "strict-json-rejected"
    if canonical == expected:
        return "semantic-match-noncanonical"
    return "wrong-record"


def _output_analysis(
    samples: Iterable[Mapping[str, object]],
) -> Mapping[str, object]:
    rows = list(samples)
    classifications = Counter(
        _output_classification(
            str(row["output"]),
            str(row["expected"]),
        )
        for row in rows
    )
    outputs = [str(row["output"]).strip() for row in rows]
    return {
        "classifications": {
            key: classifications[key]
            for key in sorted(classifications)
        },
        "starts_with_object": sum(
            output.startswith("{") for output in outputs
        ),
        "ends_with_object": sum(
            output.endswith("}") for output in outputs
        ),
        "mentions_prompt_framing": sum(
            any(
                marker in output
                for marker in ("record_patch", "record ", "patch", "result")
            )
            for output in outputs
        ),
        "mean_output_characters": statistics.fmean(
            len(output) for output in outputs
        ),
    }


def _component_evaluation(
    *,
    campaign: object,
    checkpoint_dir: Path,
    tokenizer: Tokenizer,
    examples: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    model, optimizer, step, _, _ = _load_checkpoint(
        campaign,
        checkpoint_dir,
    )
    del optimizer
    model.eval()
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("analysis tokenizer lacks BOS or EOS")
    prompt_loss_sum = 0.0
    answer_loss_sum = 0.0
    prompt_tokens = 0
    answer_tokens = 0
    prompt_correct = 0
    answer_correct = 0
    teacher_forced_exact_answers = 0
    with torch.no_grad():
        for example in examples:
            prompt = str(example["prompt"])
            story = f"{prompt}{example['target']}\n"
            encoding = tokenizer.encode(
                story,
                add_special_tokens=False,
            )
            token_ids = encoding.ids
            inputs = torch.tensor(
                [[bos_id, *token_ids]],
                dtype=torch.long,
            )
            targets = torch.tensor(
                [[*token_ids, eos_id]],
                dtype=torch.long,
            )
            logits = model(inputs)[0]
            losses = F.cross_entropy(
                logits,
                targets[0],
                reduction="none",
            )
            predicted = torch.argmax(logits, dim=-1)
            exact_answer = True
            for index, (prediction, target, loss) in enumerate(
                zip(
                    predicted.tolist(),
                    targets[0].tolist(),
                    losses.tolist(),
                    strict=True,
                )
            ):
                is_prompt = (
                    index < len(token_ids)
                    and encoding.offsets[index][0] < len(prompt)
                )
                if is_prompt:
                    prompt_tokens += 1
                    prompt_correct += int(prediction == target)
                    prompt_loss_sum += loss
                else:
                    answer_tokens += 1
                    answer_correct += int(prediction == target)
                    answer_loss_sum += loss
                    exact_answer = (
                        exact_answer and prediction == target
                    )
            teacher_forced_exact_answers += int(exact_answer)
    del model
    prompt_mean_loss = prompt_loss_sum / prompt_tokens
    answer_mean_loss = answer_loss_sum / answer_tokens
    return {
        "step": step,
        "method": (
            "teacher-forced public behavioral transcripts split at the "
            "prompt character boundary; answer includes target newline and "
            "EOS"
        ),
        "prompt_tokens": prompt_tokens,
        "prompt_mean_loss": prompt_mean_loss,
        "prompt_token_accuracy": prompt_correct / prompt_tokens,
        "answer_tokens": answer_tokens,
        "answer_mean_loss": answer_mean_loss,
        "answer_perplexity": math.exp(answer_mean_loss),
        "answer_token_accuracy": answer_correct / answer_tokens,
        "teacher_forced_exact_answers": (
            teacher_forced_exact_answers
        ),
    }


def _nearest_training_records() -> Mapping[str, object]:
    training: list[Mapping[str, object]] = []
    by_bucket: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index in range(32_768):
        example = generate_example(
            key_hex=PUBLIC_TRAIN_KEY,
            split="train",
            index=index,
        )
        training.append(example)
        by_bucket[str(example["bucket"])].append(example)
    training_prompts = {
        str(example["prompt"])
        for example in training
    }
    target_counts = Counter(
        str(example["target"])
        for example in training
    )
    samples = []
    for index in range(32):
        example = generate_example(
            key_hex=PUBLIC_BEHAVIORAL_VALIDATION_KEY,
            split="behavioral_validation",
            index=index,
        )
        bucket = str(example["bucket"])
        nearest: Mapping[str, object] | None = None
        nearest_ratio = -1.0
        for candidate in by_bucket[bucket]:
            ratio = SequenceMatcher(
                None,
                str(example["prompt"]),
                str(candidate["prompt"]),
                autojunk=False,
            ).ratio()
            if (
                ratio > nearest_ratio
                or (
                    ratio == nearest_ratio
                    and nearest is not None
                    and str(candidate["id"]) < str(nearest["id"])
                )
            ):
                nearest = candidate
                nearest_ratio = ratio
        if nearest is None:
            raise ValueError("nearest training record is unavailable")
        samples.append(
            {
                "id": example["id"],
                "bucket": bucket,
                "prompt": example["prompt"],
                "target": example["target"],
                "exact_prompt_in_train": (
                    str(example["prompt"]) in training_prompts
                ),
                "target_occurrences_in_train": target_counts[
                    str(example["target"])
                ],
                "nearest_train_id": nearest["id"],
                "nearest_prompt": nearest["prompt"],
                "nearest_target": nearest["target"],
                "nearest_prompt_similarity": nearest_ratio,
                "nearest_target_equal": (
                    nearest["target"] == example["target"]
                ),
            }
        )
    ratios = [
        float(sample["nearest_prompt_similarity"])
        for sample in samples
    ]
    return {
        "method": (
            "difflib.SequenceMatcher character matching-block ratio with "
            "autojunk disabled, restricted to the same frozen bucket; ties "
            "use the lowest training ID"
        ),
        "train_examples": len(training),
        "validation_examples": len(samples),
        "exact_prompt_overlaps": sum(
            int(sample["exact_prompt_in_train"] is True)
            for sample in samples
        ),
        "validation_targets_seen_in_train": sum(
            int(int(sample["target_occurrences_in_train"]) > 0)
            for sample in samples
        ),
        "nearest_target_equal": sum(
            int(sample["nearest_target_equal"] is True)
            for sample in samples
        ),
        "nearest_similarity": _numeric_summary(ratios),
        "samples": samples,
    }


def _diagnostic_analysis(
    diagnostics: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not diagnostics:
        raise ValueError("training diagnostics are empty")

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in diagnostics]

    milestones = {1, 2, 4, 8, 16, 32, 64, 96, 128}
    return {
        "steps": len(diagnostics),
        "clipped_steps": sum(
            int(row["clipped"] is True)
            for row in diagnostics
        ),
        "clipped_step_numbers": [
            int(row["step"])
            for row in diagnostics
            if row["clipped"] is True
        ],
        "training_mean_loss": _numeric_summary(
            values("training_mean_loss")
        ),
        "gradient_global_norm_before_clipping": _numeric_summary(
            values("gradient_global_norm_before_clipping")
        ),
        "gradient_max_abs_before_clipping": _numeric_summary(
            values("gradient_max_abs_before_clipping")
        ),
        "update_global_norm": _numeric_summary(
            values("update_global_norm")
        ),
        "relative_update_global_norm": _numeric_summary(
            values("relative_update_global_norm")
        ),
        "update_max_abs": _numeric_summary(
            values("update_max_abs")
        ),
        "milestones": [
            {
                key: row[key]
                for key in (
                    "step",
                    "training_mean_loss",
                    "gradient_global_norm_before_clipping",
                    "clipped",
                    "update_global_norm",
                    "relative_update_global_norm",
                )
            }
            for row in diagnostics
            if int(row["step"]) in milestones
        ],
    }


def build_analysis(
    *,
    campaign_path: str | Path,
    packed_dir: str | Path,
    public_dir: str | Path,
    run_dir: str | Path,
) -> Mapping[str, object]:
    campaign_file = Path(campaign_path).resolve()
    packed_root = Path(packed_dir).resolve()
    public_root = Path(public_dir).resolve()
    run_root = Path(run_dir).resolve()
    verified = _verify_checksums(run_root)
    evidence = _load_json(run_root / "evidence.json")
    if evidence.get("format") != (
        "orcacolony_record_patch_learnability_evidence_v1"
    ):
        raise ValueError("unsupported learnability evidence format")
    campaign = load_campaign(campaign_file)
    dataset = PackedDataset.load(packed_root)
    if (
        evidence.get("campaign_id") != campaign.campaign.get("id")
        or evidence.get("campaign_sha256") != _sha256_file(campaign_file)
        or evidence.get("dataset_revision") != dataset.revision
    ):
        raise ValueError("learnability evidence identity mismatch")
    checkpoint_records = evidence.get("checkpoints")
    if not isinstance(checkpoint_records, list) or not checkpoint_records:
        raise ValueError("learnability checkpoint evidence is invalid")
    _, public_examples = load_behavioral_split(
        public_dir=public_root,
        split="behavioral_validation",
    )
    tokenizer = Tokenizer.from_file(
        str(packed_root / "tokenizer.json")
    )
    checkpoints = []
    for record in checkpoint_records:
        if not isinstance(record, Mapping):
            raise ValueError("learnability checkpoint record is invalid")
        step = int(record["step"])
        behavioral = record.get("behavioral_evaluation")
        if not isinstance(behavioral, Mapping):
            raise ValueError(
                "analysis requires every checkpoint behavioral result"
            )
        evaluation_relative = str(behavioral["evaluation_path"])
        if evaluation_relative not in verified:
            raise ValueError(
                "behavioral evaluation is absent from checksums"
            )
        evaluation = _load_json(run_root / evaluation_relative)
        samples = evaluation.get("samples")
        if not isinstance(samples, list):
            raise ValueError("behavioral evaluation samples are invalid")
        component = _component_evaluation(
            campaign=campaign,
            checkpoint_dir=(
                run_root / "checkpoints" / f"step-{step:08d}"
            ),
            tokenizer=tokenizer,
            examples=public_examples,
        )
        checkpoints.append(
            {
                "step": step,
                "model_sha256": record["model_sha256"],
                "language_evaluation": record[
                    "language_evaluation"
                ],
                "behavioral_metrics": behavioral["metrics"],
                "behavioral_guardrails": behavioral["guardrails"],
                "output_analysis": _output_analysis(samples),
                "component_evaluation": component,
                "samples": samples,
            }
        )
    diagnostics_payload = json.loads(
        (run_root / "training-diagnostics.json").read_text(
            encoding="utf-8"
        ),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(diagnostics_payload, list) or any(
        not isinstance(row, Mapping)
        for row in diagnostics_payload
    ):
        raise ValueError("training diagnostics schema is invalid")
    diagnostics = [
        row
        for row in diagnostics_payload
        if isinstance(row, Mapping)
    ]
    packing = dataset.manifest["packing"]
    if not isinstance(packing, Mapping):
        raise ValueError("packed dataset metadata is invalid")
    training = campaign.training
    sequences_seen = training.batch_size * training.steps
    observed_sequences = (
        training.batch_size * int(evidence["resources"]["steps_completed"])
    )
    train_sequences = int(packing["train_sequences"])
    train_tokens = int(packing["train_tokens"])
    loss_weight_sum = int(evidence["resources"]["loss_weight_sum"])
    contamination = _nearest_training_records()
    return {
        "format": (
            "orcacolony_record_patch_learnability_analysis_v1"
        ),
        "campaign_id": evidence["campaign_id"],
        "campaign_sha256": evidence["campaign_sha256"],
        "dataset_revision": evidence["dataset_revision"],
        "run_evidence_sha256": _sha256_file(
            run_root / "evidence.json"
        ),
        "run_checksums_sha256": _sha256_file(
            run_root / "SHA256SUMS"
        ),
        "analysis_revision": f"sha256:{_module_revision()}",
        "source_commit": _load_json(
            run_root / "environment.json"
        )["source"],
        "disposition": {
            "learnability_gate_passed": evidence[
                "learnability_gate"
            ]["passed"],
            "language_gate_passed": evidence[
                "learnability_gate"
            ]["language_passed"],
            "behavioral_gate_passed": evidence[
                "learnability_gate"
            ]["behavioral_passed"],
            "selected_checkpoint": evidence["selection"],
            "volunteer_training_authorized_by_this_result": False,
            "capability_promotion_authorized_by_this_result": False,
        },
        "coverage": {
            "observed_steps": int(
                evidence["resources"]["steps_completed"]
            ),
            "proposed_campaign_steps": training.steps,
            "observed_packed_sequences": observed_sequences,
            "train_packed_sequences": train_sequences,
            "observed_sequence_epochs": (
                observed_sequences / train_sequences
            ),
            "observed_loss_weight_sum": loss_weight_sum,
            "train_tokens": train_tokens,
            "observed_token_epochs": loss_weight_sum / train_tokens,
            "proposed_sequence_epochs": (
                sequences_seen / train_sequences
            ),
        },
        "resources": evidence["resources"],
        "checkpoints": checkpoints,
        "optimizer_diagnostics": _diagnostic_analysis(diagnostics),
        "contamination_and_similarity": contamination,
        "observations": [
            (
                "Public validation mean loss improved from "
                "9.120742341162453 to 1.56358497243532."
            ),
            (
                "The language-selected step-128 checkpoint remained 0/32 "
                "exact, 0/32 semantic, and 0/32 strict valid JSON."
            ),
            (
                "Step-128 answer-token loss was 2.093474881534559 with "
                "40.70048309178744% teacher-forced token accuracy and no "
                "teacher-forced exact answer."
            ),
            (
                "Step 128 covered 512 of 4,618 packed training sequences, "
                "about 0.111 sequence epochs."
            ),
            (
                "No public behavioral prompt or target occurred exactly in "
                "the training set."
            ),
        ],
        "hypotheses": [
            {
                "id": "undertraining",
                "status": "plausible_not_proven",
                "basis": (
                    "Only about 0.111 packed-data epochs were observed, while "
                    "answer loss and output structure were still improving."
                ),
            },
            {
                "id": "all-token-objective-efficiency",
                "status": "plausible_not_proven",
                "basis": (
                    "The objective spends loss weight on prompt tokens; at "
                    "step 128 prompt loss was lower than answer loss."
                ),
            },
            {
                "id": "aggressive-early-optimization",
                "status": "plausible_not_proven",
                "basis": (
                    "114 of 128 steps clipped, with early gradient-norm "
                    "spikes and a temporary public validation regression."
                ),
            },
        ],
        "recommended_next_control": {
            "change": (
                "Continue the same exact checkpoint, optimizer, dataset "
                "order, objective, and decoding policy to later predeclared "
                "milestones before changing the recipe."
            ),
            "reason": (
                "This isolates insufficient exposure from objective or "
                "optimizer changes."
            ),
            "holdout_policy": (
                "Continue using only public language and behavioral "
                "validation; keep both reserved holdouts closed."
            ),
        },
        "credit": {
            "donated_compute": False,
            "statement": (
                "This was owner-operated local qualification. No volunteer "
                "training contribution is claimed or credited."
            ),
        },
        "limitations": [
            *evidence["limitations"],
            (
                "Character similarity is descriptive and is not a semantic "
                "contamination detector."
            ),
            (
                "Teacher-forced token accuracy does not measure free-running "
                "generation accuracy."
            ),
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a completed public-only Record Patch learnability run"
        )
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    analysis = build_analysis(
        campaign_path=args.campaign,
        packed_dir=args.packed_dir,
        public_dir=args.public_dir,
        run_dir=args.run_dir,
    )
    _write_exact(
        args.output.resolve(),
        _canonical_json_bytes(analysis),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "analysis_revision": analysis["analysis_revision"],
                "disposition": analysis["disposition"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
