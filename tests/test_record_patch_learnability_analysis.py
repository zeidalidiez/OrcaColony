import hashlib
from pathlib import Path

import pytest

from orcacolony.record_patch_learnability_analysis import (
    _build_parser,
    _numeric_summary,
    _output_classification,
    _verify_checksums,
)


@pytest.mark.parametrize(
    ("output", "expected", "classification"),
    (
        ('{"a":1}', '{"a":1}', "exact"),
        ("", '{"a":1}', "empty"),
        ('{"a":1,"a":2}', '{"a":1}', "duplicate-object-keys"),
        ('{"a":', '{"a":1}', "malformed-json"),
        ('{"a":1} extra', '{"a":1}', "trailing-content"),
        ("1", '{"a":1}', "non-object-json"),
        ('{ "a": 1 }', '{"a":1}', "semantic-match-noncanonical"),
        ('{"a":2}', '{"a":1}', "wrong-record"),
    ),
)
def test_output_classification_preserves_failure_modes(
    output: str,
    expected: str,
    classification: str,
) -> None:
    assert _output_classification(output, expected) == classification


def test_numeric_summary_is_explicit_and_finite() -> None:
    assert _numeric_summary([1.0, 2.0, 3.0, 4.0]) == {
        "min": 1.0,
        "p25": 2.0,
        "median": 2.5,
        "p75": 3.0,
        "max": 4.0,
        "mean": 2.5,
    }

    with pytest.raises(ValueError, match="finite and nonempty"):
        _numeric_summary([])


def test_analysis_checksum_verification_rejects_mismatch(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  artifact.json\n",
        encoding="utf-8",
    )

    assert _verify_checksums(tmp_path) == {
        "artifact.json": digest,
    }

    artifact.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_checksums(tmp_path)


def test_analysis_cli_has_no_private_holdout_argument() -> None:
    destinations = {
        action.dest
        for action in _build_parser()._actions
    }

    assert "final_holdout" not in destinations
    assert "holdout_key" not in destinations
