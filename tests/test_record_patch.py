import json
from pathlib import Path

import pytest

from orcacolony.record_patch import (
    BEHAVIORAL_BUCKETS,
    PUBLIC_BEHAVIORAL_VALIDATION_KEY,
    T2_PARAMETER_COUNT,
    apply_patch,
    evaluate_predictions,
    freeze_record_patch,
    generate_example,
    load_behavioral_split,
    write_oracle_predictions,
)
from orcacolony.reference import load_campaign


def test_record_patch_examples_are_deterministic_and_oracle_checked() -> None:
    first = [
        generate_example(
            key_hex=PUBLIC_BEHAVIORAL_VALIDATION_KEY,
            split="behavioral_validation",
            index=index,
        )
        for index in range(len(BEHAVIORAL_BUCKETS))
    ]
    second = [
        generate_example(
            key_hex=PUBLIC_BEHAVIORAL_VALIDATION_KEY,
            split="behavioral_validation",
            index=index,
        )
        for index in range(len(BEHAVIORAL_BUCKETS))
    ]

    assert first == second
    assert [example["bucket"] for example in first] == list(
        BEHAVIORAL_BUCKETS
    )
    for example in first:
        expected = apply_patch(
            example["record"],
            example["operations"],
        )
        assert json.loads(example["target"]) == expected


def _freeze(tmp_path: Path):
    return freeze_record_patch(
        public_dir=tmp_path / "public",
        private_dir=tmp_path / "private",
        campaign_path=tmp_path / "campaign.json",
        train_examples=16,
        language_validation_examples=16,
        behavioral_validation_examples=8,
        behavioral_final_holdout_examples=8,
        steps=2,
    )


def test_record_patch_freeze_separates_holdout_and_builds_true_t2(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path)
    repeated = _freeze(tmp_path)

    assert repeated == frozen
    assert not (
        frozen.public_dir / "behavioral-final-holdout.jsonl"
    ).exists()
    assert (
        frozen.private_dir / "behavioral-final-holdout.jsonl"
    ).is_file()
    assert (
        frozen.private_dir / "holdout-key.json"
    ).stat().st_mode & 0o077 == 0
    campaign = load_campaign(frozen.campaign_path)
    assert campaign.model.parameters == T2_PARAMETER_COUNT
    assert campaign.model.layers == 8
    assert campaign.model.width == 384
    assert campaign.publication == {
        "format": "orcacolony_huggingface_publication_v1",
        "model_repo_id": "OrcaColony/record-patch-t2-v1",
        "dataset_repo_id": "OrcaColony/record-patch-v1",
        "model_license": "apache-2.0",
        "dataset_license": "cc0-1.0",
        "visibility_policy": "private_review_then_public",
    }
    _, validation = load_behavioral_split(
        public_dir=frozen.public_dir,
        split="behavioral_validation",
    )
    _, holdout = load_behavioral_split(
        public_dir=frozen.public_dir,
        split="behavioral_final_holdout",
        final_holdout_path=(
            frozen.private_dir / "behavioral-final-holdout.jsonl"
        ),
    )
    assert {row["prompt"] for row in validation}.isdisjoint(
        {row["prompt"] for row in holdout}
    )


def test_record_patch_evaluator_scores_oracle_and_rejects_missing_ids(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path)
    predictions = tmp_path / "oracle.jsonl"
    write_oracle_predictions(
        public_dir=frozen.public_dir,
        split="behavioral_validation",
        output_path=predictions,
    )

    result = evaluate_predictions(
        public_dir=frozen.public_dir,
        split="behavioral_validation",
        predictions_path=predictions,
    )
    assert result["metrics"]["record_exact_match"] == 1.0
    assert result["metrics"]["valid_json"] == 1.0
    assert all(guardrail["passed"] for guardrail in result["guardrails"])

    rows = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction IDs differ"):
        evaluate_predictions(
            public_dir=frozen.public_dir,
            split="behavioral_validation",
            predictions_path=predictions,
        )
