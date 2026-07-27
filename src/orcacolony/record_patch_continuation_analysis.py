from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from tokenizers import Tokenizer

from .artifacts import PackedDataset
from .record_patch import load_behavioral_split
from .record_patch_learnability import (
    _canonical_json_bytes,
    _sha256_file,
    _write_exact,
)
from .record_patch_learnability_analysis import (
    _component_evaluation,
    _nearest_training_records,
    _numeric_summary,
    _output_analysis,
    _verify_checksums,
)
from .reference import load_campaign


def _strict_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _conditioning_analysis(
    samples: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    by_id = {str(example["id"]): example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError("public examples contain duplicate IDs")
    if {str(sample["id"]) for sample in samples} != set(by_id):
        raise ValueError("behavioral samples differ from public examples")

    valid_objects = 0
    expected_keys = 0
    expected_keys_present = 0
    expected_pairs = 0
    expected_pairs_matched = 0
    output_keys = 0
    output_keys_expected = 0
    unchanged_pairs = 0
    unchanged_pairs_retained = 0
    changed_fields = 0
    changed_field_endpoints_matched = 0
    input_pairs_in_output = 0
    output_pairs = 0
    all_changed_endpoints_matched = 0
    valid_outputs: list[str] = []
    output_key_counts: Counter[str] = Counter()
    output_value_counts: Counter[str] = Counter()
    missing = object()

    for sample in samples:
        example = by_id[str(sample["id"])]
        record = example["record"]
        if not isinstance(record, Mapping):
            raise ValueError("public input record is invalid")
        expected = json.loads(str(example["target"]))
        if not isinstance(expected, Mapping):
            raise ValueError("public expected record is invalid")
        parsed = sample.get("parsed_output")
        prediction = (
            parsed
            if sample.get("valid_json") is True
            and isinstance(parsed, Mapping)
            else {}
        )
        is_valid_object = bool(prediction) or (
            sample.get("valid_json") is True
            and isinstance(parsed, Mapping)
        )
        valid_objects += int(is_valid_object)
        if is_valid_object:
            valid_outputs.append(str(sample["output"]))

        expected_keys += len(expected)
        expected_pairs += len(expected)
        expected_keys_present += sum(
            int(key in prediction) for key in expected
        )
        expected_pairs_matched += sum(
            int(
                key in prediction
                and _strict_equal(prediction[key], value)
            )
            for key, value in expected.items()
        )
        output_keys += len(prediction)
        output_pairs += len(prediction)
        output_keys_expected += sum(
            int(key in expected) for key in prediction
        )
        input_pairs_in_output += sum(
            int(
                key in record
                and _strict_equal(record[key], value)
            )
            for key, value in prediction.items()
        )
        output_key_counts.update(str(key) for key in prediction)
        output_value_counts.update(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for value in prediction.values()
        )

        fields = set(record) | set(expected)
        changed = {
            key
            for key in fields
            if (
                key not in record
                or key not in expected
                or not _strict_equal(record[key], expected[key])
            )
        }
        unchanged = fields - changed
        unchanged_pairs += len(unchanged)
        unchanged_pairs_retained += sum(
            int(
                key in prediction
                and _strict_equal(prediction[key], expected[key])
            )
            for key in unchanged
        )
        changed_fields += len(changed)

        def endpoint_matches(key: str) -> bool:
            predicted = prediction.get(key, missing)
            target = expected.get(key, missing)
            if predicted is missing or target is missing:
                return predicted is target
            return _strict_equal(predicted, target)

        endpoint_results = [
            endpoint_matches(str(key))
            for key in changed
        ]
        changed_field_endpoints_matched += sum(
            int(result) for result in endpoint_results
        )
        all_changed_endpoints_matched += int(
            is_valid_object
            and bool(endpoint_results)
            and all(endpoint_results)
        )

    return {
        "scope": (
            "all public behavioral examples; invalid outputs contribute "
            "zero matched fields"
        ),
        "examples": len(samples),
        "valid_object_examples": valid_objects,
        "unique_outputs": len(
            {str(sample["output"]) for sample in samples}
        ),
        "unique_valid_outputs": len(set(valid_outputs)),
        "expected_key_recall": {
            "matched": expected_keys_present,
            "total": expected_keys,
            "ratio": _ratio(expected_keys_present, expected_keys),
        },
        "output_key_precision": {
            "matched": output_keys_expected,
            "total": output_keys,
            "ratio": _ratio(output_keys_expected, output_keys),
        },
        "expected_key_value_recall": {
            "matched": expected_pairs_matched,
            "total": expected_pairs,
            "ratio": _ratio(expected_pairs_matched, expected_pairs),
        },
        "unchanged_key_value_retention": {
            "matched": unchanged_pairs_retained,
            "total": unchanged_pairs,
            "ratio": _ratio(
                unchanged_pairs_retained,
                unchanged_pairs,
            ),
        },
        "changed_field_endpoint_match": {
            "matched": changed_field_endpoints_matched,
            "total": changed_fields,
            "ratio": _ratio(
                changed_field_endpoints_matched,
                changed_fields,
            ),
            "examples_with_all_changed_endpoints_matched": (
                all_changed_endpoints_matched
            ),
            "warning": (
                "A missing deleted key can match its endpoint even when "
                "the rest of the record is wrong. This is diagnostic only."
            ),
        },
        "output_key_value_pairs_copied_from_input": {
            "matched": input_pairs_in_output,
            "total": output_pairs,
            "ratio": _ratio(input_pairs_in_output, output_pairs),
        },
        "most_common_output_keys": [
            {"key": key, "count": count}
            for key, count in output_key_counts.most_common(12)
        ],
        "most_common_output_values": [
            {"value": value, "count": count}
            for value, count in output_value_counts.most_common(12)
        ],
    }


def _optimizer_analysis(
    diagnostics: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not diagnostics:
        raise ValueError("continuation diagnostics are empty")
    steps = [int(row["step"]) for row in diagnostics]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("continuation diagnostics are not contiguous")

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in diagnostics]

    milestones = {129, 160, 192, 224, 256, 320, 384, 448, 512}
    clipped_steps = sum(
        int(row["clipped"] is True) for row in diagnostics
    )
    return {
        "steps": len(diagnostics),
        "first_step": steps[0],
        "last_step": steps[-1],
        "clipped_steps": clipped_steps,
        "clipped_step_ratio": clipped_steps / len(diagnostics),
        "training_mean_loss": _numeric_summary(
            values("training_mean_loss")
        ),
        "gradient_global_norm_before_clipping": _numeric_summary(
            values("gradient_global_norm_before_clipping")
        ),
        "update_global_norm": _numeric_summary(
            values("update_global_norm")
        ),
        "relative_update_global_norm": _numeric_summary(
            values("relative_update_global_norm")
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


def _simple_benchmarks(
    examples: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    identity_matches = 0
    empty_matches = 0
    for example in examples:
        target = str(example["target"])
        identity = json.dumps(
            example["record"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        identity_matches += int(identity == target)
        empty_matches += int(target == "{}")
    count = len(examples)
    return {
        "examples": count,
        "copy_input_record": {
            "exact_matches": identity_matches,
            "ratio": identity_matches / count,
            "description": (
                "Canonicalize and return the input record without applying "
                "the patch."
            ),
        },
        "empty_object": {
            "exact_matches": empty_matches,
            "ratio": empty_matches / count,
            "description": "Return an empty JSON object.",
        },
        "deterministic_task_oracle": {
            "exact_matches": count,
            "ratio": 1.0,
            "description": (
                "Apply the declared Record Patch semantics and canonicalize "
                "the resulting flat JSON record."
            ),
        },
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _module_revision() -> str:
    return _sha256_file(Path(__file__).resolve())


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
    if (
        evidence.get("format")
        != "orcacolony_record_patch_continuation_evidence_v1"
    ):
        raise ValueError("unsupported continuation evidence format")
    campaign = load_campaign(campaign_file)
    dataset = PackedDataset.load(packed_root)
    if (
        evidence.get("campaign_id") != campaign.campaign.get("id")
        or evidence.get("campaign_sha256") != _sha256_file(campaign_file)
        or evidence.get("dataset_revision") != dataset.revision
    ):
        raise ValueError("continuation evidence identity mismatch")
    checkpoint_records = evidence.get("checkpoints")
    if not isinstance(checkpoint_records, list) or not checkpoint_records:
        raise ValueError("continuation checkpoint evidence is invalid")
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
            raise ValueError("continuation checkpoint record is invalid")
        step = int(record["step"])
        behavioral = record.get("behavioral_evaluation")
        if not isinstance(behavioral, Mapping):
            raise ValueError("continuation behavioral evidence is invalid")
        evaluation_relative = str(behavioral["evaluation_path"])
        if evaluation_relative not in verified:
            raise ValueError(
                "behavioral evaluation is absent from checksums"
            )
        evaluation = _load_json(run_root / evaluation_relative)
        samples = evaluation.get("samples")
        if not isinstance(samples, list) or any(
            not isinstance(sample, Mapping) for sample in samples
        ):
            raise ValueError("behavioral evaluation samples are invalid")
        typed_samples = [
            sample
            for sample in samples
            if isinstance(sample, Mapping)
        ]
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
                "optimizer_sha256": record["optimizer_sha256"],
                "language_evaluation": record[
                    "language_evaluation"
                ],
                "behavioral_metrics": behavioral["metrics"],
                "behavioral_guardrails": behavioral["guardrails"],
                "output_analysis": _output_analysis(typed_samples),
                "conditioning_analysis": _conditioning_analysis(
                    typed_samples,
                    public_examples,
                ),
                "component_evaluation": component,
                "samples": typed_samples,
            }
        )
    diagnostics_payload = json.loads(
        (run_root / "training-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(diagnostics_payload, list) or any(
        not isinstance(row, Mapping)
        for row in diagnostics_payload
    ):
        raise ValueError("continuation diagnostics schema is invalid")
    diagnostics = [
        row
        for row in diagnostics_payload
        if isinstance(row, Mapping)
    ]
    packing = dataset.manifest["packing"]
    if not isinstance(packing, Mapping):
        raise ValueError("packed dataset metadata is invalid")
    resources = evidence["resources"]
    if not isinstance(resources, Mapping):
        raise ValueError("continuation resource evidence is invalid")
    training = campaign.training
    final_step = int(resources["final_step"])
    resume_step = int(resources["resume_step"])
    train_sequences = int(packing["train_sequences"])
    train_tokens = int(packing["train_tokens"])
    total_sequences = training.batch_size * final_step
    continuation_sequences = (
        training.batch_size * (final_step - resume_step)
    )
    total_loss_weight = int(
        resources["total_trajectory_loss_weight_sum"]
    )
    checkpoints_by_step = {
        int(checkpoint["step"]): checkpoint
        for checkpoint in checkpoints
    }
    selected_step = int(evidence["selection"]["step"])
    selected = checkpoints_by_step[selected_step]
    parent = checkpoints_by_step[resume_step]
    selected_conditioning = selected["conditioning_analysis"]
    if not isinstance(selected_conditioning, Mapping):
        raise ValueError("selected conditioning analysis is invalid")
    selected_pair_recall = selected_conditioning[
        "expected_key_value_recall"
    ]
    if not isinstance(selected_pair_recall, Mapping):
        raise ValueError("selected pair-recall analysis is invalid")
    contamination = _nearest_training_records()
    return {
        "format": "orcacolony_record_patch_continuation_analysis_v1",
        "reporting_position": {
            "author_type": "software_agent",
            "statement": (
                "These are the agent's findings from the bounded local "
                "continuation. The frozen benchmark data, evaluator, model "
                "revision, and public scores are the reproducible record."
            ),
        },
        "benchmark": {
            "id": "record-patch-v1",
            "use_case": (
                "Apply ordered SET, DELETE, and RENAME operations to a flat "
                "JSON record and emit the exact canonical result."
            ),
            "claim_scope": (
                "Performance on this task only. The benchmark does not "
                "measure or imply general intelligence."
            ),
            "primary_metric": "record exact match",
            "public_examples": len(public_examples),
            "simple_reference_scores": _simple_benchmarks(public_examples),
        },
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
            "benchmark_gate_passed": evidence[
                "learnability_gate"
            ]["passed"],
            "language_diagnostic_passed": evidence[
                "learnability_gate"
            ]["language_passed"],
            "record_exact_match_gate_passed": evidence[
                "learnability_gate"
            ]["behavioral_passed"],
            "selected_checkpoint": evidence["selection"],
            "narrow_task_claim_supported": False,
            "general_capability_claim_evaluated": False,
            "community_campaign_authorized_by_this_result": False,
            "model_promotion_authorized_by_this_result": False,
        },
        "coverage": {
            "resume_step": resume_step,
            "final_step": final_step,
            "continuation_steps": final_step - resume_step,
            "proposed_campaign_steps": training.steps,
            "continuation_packed_sequences": continuation_sequences,
            "total_packed_sequences_seen": total_sequences,
            "train_packed_sequences": train_sequences,
            "total_sequence_epochs": total_sequences / train_sequences,
            "total_loss_weight_sum": total_loss_weight,
            "train_tokens": train_tokens,
            "total_token_epochs": total_loss_weight / train_tokens,
            "proposed_sequence_epochs": (
                training.batch_size
                * training.steps
                / train_sequences
            ),
        },
        "resources": resources,
        "checkpoints": checkpoints,
        "optimizer_diagnostics": _optimizer_analysis(diagnostics),
        "contamination_and_similarity": contamination,
        "observations": [
            (
                "Public language mean loss fell from "
                f"{parent['language_evaluation']['mean_loss']} at step "
                f"{resume_step} to "
                f"{selected['language_evaluation']['mean_loss']} at step "
                f"{selected_step}."
            ),
            (
                "Strict canonical JSON rose from "
                f"{parent['behavioral_metrics']['canonical_json']} at step "
                f"{resume_step} to "
                f"{selected['behavioral_metrics']['canonical_json']} at "
                f"step {selected_step}."
            ),
            (
                f"The selected checkpoint remained "
                f"{selected['behavioral_metrics']['record_exact_match_count']}"
                f"/{selected['behavioral_metrics']['examples']} exact and "
                f"{selected['behavioral_metrics']['semantic_match']} "
                "semantic match."
            ),
            (
                "Across all public examples, the selected checkpoint "
                f"reproduced {selected_pair_recall['matched']} of "
                f"{selected_pair_recall['total']} expected key-value pairs."
            ),
            (
                "Teacher-forced answer-token accuracy improved, but no "
                "checkpoint produced a complete teacher-forced answer."
            ),
            (
                f"The complete trajectory covered "
                f"{total_sequences}/{train_sequences} packed sequences, "
                f"or {total_sequences / train_sequences:.3f} epochs."
            ),
        ],
        "hypotheses": [
            {
                "id": "syntax-before-task-conditioning",
                "status": "observed_pattern",
                "basis": (
                    "Canonical JSON reached 30/32 while exact and semantic "
                    "record matches stayed at zero. Expected key-value recall "
                    "was also low."
                ),
            },
            {
                "id": "all-token-objective-efficiency",
                "status": "plausible_not_proven",
                "basis": (
                    "Prompt loss continued to improve through step 512 while "
                    "answer loss was best at step 256 and no complete answer "
                    "was correct under teacher forcing."
                ),
            },
            {
                "id": "insufficient-exposure",
                "status": "still_possible_not_isolated",
                "basis": (
                    "The trajectory covered less than half of one packed-data "
                    "epoch, but added exposure improved formatting without "
                    "producing a task-correct example."
                ),
            },
        ],
        "recommended_next_control": {
            "change": (
                "Add an explicit answer-token mask and compare an answer-only "
                "causal objective from the same initialization for 512 steps."
            ),
            "constants": (
                "Keep the model, tokenizer, examples, order, batch size, "
                "optimizer, learning rate, decoding, public milestones, and "
                "task evaluator fixed."
            ),
            "reason": (
                "This tests whether spending loss weight on the prompt is "
                "preventing efficient learning of the output transformation."
            ),
            "holdout_policy": (
                "Use only public validation and keep both reserved final "
                "holdouts closed."
            ),
        },
        "credit": {
            "community_campaign": False,
            "donated_compute": False,
            "statement": (
                "This was owner-operated local benchmark qualification, not "
                "a community campaign. No community compute contribution is "
                "claimed in this result."
            ),
        },
        "holdout": evidence["holdout"],
        "limitations": [
            *evidence["limitations"],
            (
                "The 32-example public suite gives a coarse task estimate and "
                "has been used repeatedly for recipe decisions."
            ),
            (
                "Teacher-forced token accuracy is diagnostic and does not "
                "replace free-running exact task evaluation."
            ),
            (
                "The simple reference scores are derived directly from the "
                "public task records; Hugging Face publication should retain "
                "the runnable evaluator and exact revisions."
            ),
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the public-only Record Patch continuation as a narrow "
            "task benchmark"
        )
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    analysis = build_analysis(
        campaign_path=args.campaign,
        packed_dir=args.packed_dir,
        public_dir=args.public_dir,
        run_dir=args.run,
    )
    _write_exact(args.output.resolve(), _canonical_json_bytes(analysis))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "analysis_revision": analysis["analysis_revision"],
                "disposition": analysis["disposition"],
                "holdout": analysis["holdout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
