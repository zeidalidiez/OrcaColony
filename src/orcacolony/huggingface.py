from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from .auxiliary_contributions import (
    validate_auxiliary_contribution_snapshot,
    verify_public_auxiliary_snapshot_artifacts,
)


_REPO_ID = re.compile(
    r"OrcaColony/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\Z"
)
_LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload), encoding="utf-8", newline="\n")


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, label: str) -> Mapping[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate Hugging Face package JSON key: {key}")
        payload[key] = value
    return payload


def _safe_relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or value.startswith(("/", "\\"))
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is unsafe")
    return path


def _regular_file_map(root: Path, label: str) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    files: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"{label} may not contain symlinks: {relative}")
        if candidate.is_file():
            files[relative] = candidate
        elif not candidate.is_dir():
            raise ValueError(f"{label} contains a non-regular entry: {relative}")
    return files


def _copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"package input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def verify_release_bundle(release_dir: str | Path) -> Mapping[str, object]:
    root = Path(release_dir).resolve()
    manifest = _load_mapping(root / "release-manifest.json", "release manifest")
    if manifest.get("format") != "orcacolony_release_bundle_v1":
        raise ValueError("unsupported OrcaColony release bundle")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("release manifest file map is missing")
    actual_files = _regular_file_map(root, "release bundle")
    expected_file_names = set(files) | {
        "release-manifest.json",
        "SHA256SUMS",
    }
    if set(actual_files) != expected_file_names:
        raise ValueError("release bundle contains an unmanifested or missing file")
    for raw_path, expected in files.items():
        path = _safe_relative_path(raw_path, "release file")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("release manifest contains an invalid digest")
        candidate = actual_files[path.as_posix()]
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"release file is missing: {path.as_posix()}")
        if _sha256_file(candidate) != expected:
            raise ValueError(f"release file digest mismatch: {path.as_posix()}")
    release_manifest_sha256 = _sha256_file(root / "release-manifest.json")
    expected_checksums = "".join(
        f"{files[name]}  {name}\n" for name in sorted(files)
    ) + (
        f"{release_manifest_sha256}  release-manifest.json\n"
    )
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("release SHA256SUMS differs from its manifest")
    return manifest


def _validate_repo_id(value: str, label: str) -> str:
    if (
        _REPO_ID.fullmatch(value) is None
        or "--" in value
        or ".." in value
    ):
        raise ValueError(
            f"{label} must use the OrcaColony organization namespace"
        )
    return value


def _validate_license(value: str, label: str) -> str:
    if (
        _LICENSE_ID.fullmatch(value) is None
        or value.casefold()
        in {"choose-explicitly", "other", "replace-me", "tbd", "unknown"}
        or value.casefold().startswith("replace-")
    ):
        raise ValueError(
            f"{label} must be an explicit Hugging Face license identifier"
        )
    return value


def _validate_https_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in value)
        or any(character in "<>\"'()[]" for character in value)
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return value


def _card_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _card_attribution(
    attribution: Mapping[str, object],
) -> tuple[int, int, int, int]:
    totals = attribution.get("all_contributions")
    contributors = attribution.get("public_contributors")
    anonymous = attribution.get("anonymous_contributors")
    if (
        attribution.get("format") != "orcacolony_attribution_snapshot_v1"
        or not isinstance(totals, Mapping)
        or not isinstance(contributors, list)
        or any(not isinstance(item, Mapping) for item in contributors)
        or not isinstance(anonymous, Mapping)
    ):
        raise ValueError("release attribution snapshot is invalid")
    return (
        _card_count(
            totals.get("accepted_assignments"),
            "attribution accepted assignments",
        ),
        _card_count(
            totals.get("accepted_tokens"),
            "attribution accepted tokens",
        ),
        len(contributors),
        _card_count(
            anonymous.get("count"),
            "attribution anonymous contributor count",
        ),
    )


def _card_auxiliary_contributions(
    auxiliary: Mapping[str, object],
) -> tuple[str, int, int, int, int]:
    status = auxiliary.get("record_status")
    totals = auxiliary.get("all_contributions")
    contributors = auxiliary.get("public_contributors")
    anonymous = auxiliary.get("anonymous_contributors")
    if (
        auxiliary.get("format")
        != "orcacolony_auxiliary_contribution_snapshot_v1"
        or status not in {"not_supplied", "owner_reviewed"}
        or not isinstance(totals, Mapping)
        or not isinstance(contributors, list)
        or any(not isinstance(item, Mapping) for item in contributors)
        or not isinstance(anonymous, Mapping)
    ):
        raise ValueError("release auxiliary contribution snapshot is invalid")
    contributor_count = _card_count(
        totals.get("contributor_count"),
        "auxiliary contributor count",
    )
    contribution_count = _card_count(
        totals.get("contribution_count"),
        "auxiliary contribution count",
    )
    anonymous_count = _card_count(
        anonymous.get("count"),
        "anonymous auxiliary contributor count",
    )
    if contributor_count != len(contributors) + anonymous_count:
        raise ValueError(
            "auxiliary contributor total differs from its public and anonymous counts"
        )
    if status == "not_supplied" and (
        contributor_count != 0
        or contribution_count != 0
        or auxiliary.get("source_ledger_sha256") is not None
    ):
        raise ValueError(
            "unsupplied auxiliary contribution record contains contribution data"
        )
    return (
        str(status),
        contributor_count,
        contribution_count,
        len(contributors),
        anonymous_count,
    )


def _campaign_evaluation_markdown(
    evaluation: Mapping[str, object] | None,
) -> str:
    if evaluation is None:
        return "No campaign-owner evaluation evidence was supplied."
    lines = [
        "The package includes the campaign-owner evaluation record.",
    ]
    comparisons = evaluation.get("comparisons")
    if isinstance(comparisons, list):
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                continue
            summary = comparison.get("summary")
            if isinstance(summary, str) and summary.strip():
                lines.append(f"- Comparison: {summary.strip()}")
            metrics = comparison.get("metrics")
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, Mapping):
                    continue
                label = metric.get("label")
                unit = metric.get("unit")
                baseline = metric.get("baseline_value")
                candidate = metric.get("candidate_value")
                change = metric.get("absolute_change")
                direction = metric.get("direction")
                if (
                    isinstance(label, str)
                    and isinstance(unit, str)
                    and isinstance(direction, str)
                    and isinstance(baseline, (int, float))
                    and not isinstance(baseline, bool)
                    and isinstance(candidate, (int, float))
                    and not isinstance(candidate, bool)
                    and isinstance(change, (int, float))
                    and not isinstance(change, bool)
                ):
                    lines.append(
                        f"  - {label.strip()}: `{baseline}` to `{candidate}` "
                        f"{unit.strip()} (raw change `{change}`; owner-declared "
                        f"direction `{direction}`)."
                    )
    findings = evaluation.get("findings")
    if isinstance(findings, list) and findings:
        lines.append("- Recorded findings:")
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            label = finding.get("label")
            kind = finding.get("kind")
            description = finding.get("description")
            if all(
                isinstance(value, str) and value.strip()
                for value in (label, kind, description)
            ):
                lines.append(
                    f"  - {str(label).strip()} ({str(kind).strip()}): "
                    f"{str(description).strip()}"
                )
    return "\n".join(lines)


def _model_card(
    *,
    campaign_id: str,
    model_repo_id: str,
    dataset_repo_id: str,
    model_license: str,
    dataset_license: str,
    release_classification: str,
    source_repository: str,
    source_revision: str,
    has_language_model_holdout: bool,
    campaign: Mapping[str, object],
    campaign_lock: Mapping[str, object],
    dataset_manifest: Mapping[str, object],
    checkpoint: Mapping[str, object],
    attribution: Mapping[str, object],
    auxiliary_contributions: Mapping[str, object],
    campaign_evaluation: Mapping[str, object] | None,
    promotion_evidence: Mapping[str, object] | None,
) -> str:
    campaign_metadata = campaign.get("campaign")
    model = campaign.get("model")
    training = campaign.get("training")
    dataset_source = dataset_manifest.get("source")
    if (
        not isinstance(campaign_metadata, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(dataset_source, Mapping)
    ):
        raise ValueError("release card metadata is invalid")
    accepted_assignments, accepted_tokens, public_count, anonymous_count = (
        _card_attribution(attribution)
    )
    (
        auxiliary_status,
        auxiliary_contributor_count,
        auxiliary_contribution_count,
        auxiliary_public_count,
        auxiliary_anonymous_count,
    ) = _card_auxiliary_contributions(auxiliary_contributions)
    research = campaign.get("research")
    if (
        isinstance(research, Mapping)
        and research.get("format") == "orcacolony_campaign_research_v2"
    ):
        question = research.get("question")
        usage_scenario = research.get("usage_scenario")
        claim = (
            f"Usage scenario: {usage_scenario} Research question: {question}"
            if isinstance(question, str) and isinstance(usage_scenario, str)
            else "Review the campaign's declared research contract."
        )
    else:
        claim = (
            research.get("claim")
            if isinstance(research, Mapping)
            and isinstance(research.get("claim"), str)
            else "Reproduce and inspect the declared OrcaColony training system."
        )
    selected_evaluation = checkpoint.get("evaluation")
    evaluation_text = (
        (
            f"Built-in training diagnostic at the released checkpoint: "
            f"validation mean loss "
            f"`{selected_evaluation.get('mean_loss')}` at step "
            f"`{selected_evaluation.get('step')}`."
            if isinstance(research, Mapping)
            and research.get("format") == "orcacolony_campaign_research_v2"
            else (
                f"Selected validation mean loss: "
                f"`{selected_evaluation.get('mean_loss')}` at step "
                f"`{selected_evaluation.get('step')}`."
            )
        )
        if isinstance(selected_evaluation, Mapping)
        else "No built-in validation diagnostic was selected for this release."
    )
    campaign_evaluation_text = _campaign_evaluation_markdown(
        campaign_evaluation
    )
    initial_identity = (
        campaign_lock.get("base_model_sha256")
        or campaign_lock.get("checkpoint_sha256")
    )
    limitations = (
        campaign_evaluation.get("limitations")
        if isinstance(campaign_evaluation, Mapping)
        else (
            promotion_evidence.get("limitations")
            if isinstance(promotion_evidence, Mapping)
            else None
        )
    )
    limitation_lines = (
        "".join(f"- {item}\n" for item in limitations)
        if isinstance(limitations, list)
        and limitations
        and all(isinstance(item, str) and item.strip() for item in limitations)
        else (
            "- No campaign-specific limitation list was supplied with this "
            "checkpoint release.\n"
        )
    )
    if release_classification == "campaign_result":
        disposition_text = (
            "This package includes the campaign owner's evaluation evidence and "
            "computed metric comparisons. The framework does not assign a pass "
            "or promotion decision."
        )
    elif release_classification == "campaign_checkpoint":
        disposition_text = (
            "This package contains a campaign checkpoint without a supplied "
            "campaign-specific evaluation record."
        )
    elif release_classification == "capability_model":
        disposition_text = (
            "This historical package includes a release-time language-model "
            "holdout diagnostic and a legacy passing promotion record."
        )
    elif has_language_model_holdout:
        disposition_text = (
            "This historical package includes a reserved language-model holdout "
            "result but no passing legacy promotion record."
        )
    else:
        disposition_text = "This is a systems-evidence checkpoint."
    if auxiliary_status == "owner_reviewed":
        auxiliary_credit_text = (
            f"The owner-reviewed auxiliary record contains "
            f"`{auxiliary_contribution_count}` contribution(s) from "
            f"`{auxiliary_contributor_count}` contributor(s). "
            f"`{auxiliary_public_count}` chose named or pseudonymous credit and "
            f"`{auxiliary_anonymous_count}` chose anonymous credit."
        )
    else:
        auxiliary_credit_text = (
            "No owner-reviewed auxiliary contribution record was supplied. "
            "This private review package is not ready for public publication."
        )
    return (
        "---\n"
        "library_name: orcacolony\n"
        f"license: {model_license}\n"
        "pipeline_tag: text-generation\n"
        "datasets:\n"
        f"- {dataset_repo_id}\n"
        "tags:\n"
        "- orcacolony\n"
        "- community-training\n"
        "- research\n"
        "---\n\n"
        f"# {campaign_id}\n\n"
        f"Repository target: `{model_repo_id}`.\n\n"
        "## Release summary\n\n"
        f"Claim: {claim}\n\n"
        f"Release classification: `{release_classification}`. {disposition_text}\n\n"
        f"- Architecture: `{model.get('architecture')}` with "
        f"`{model.get('parameters')}` parameters.\n"
        f"- Objective: `{campaign_metadata.get('objective')}` with "
        f"`{campaign_metadata.get('loss_mask')}`.\n"
        f"- Training: `{training.get('steps')}` configured optimizer steps; "
        f"learning rate `{training.get('learning_rate')}`; seed "
        f"`{training.get('seed')}`.\n"
        f"- Initialization/base identity: `{initial_identity}`.\n"
        f"- Selected checkpoint step: `{checkpoint.get('step')}`.\n"
        f"- Training data: `{dataset_source.get('dataset')}` at revision "
        f"`{dataset_source.get('revision')}` in "
        f"[`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id}).\n"
        f"- Licenses: model `{model_license}`; dataset "
        f"`{dataset_license}`.\n\n"
        "## Evaluation\n\n"
        f"{evaluation_text}\n\n"
        f"{campaign_evaluation_text}\n\n"
        "Training diagnostics and the campaign owner's usage evaluation are "
        "separate records. Review the declared evaluator, inputs, sample-level "
        "artifacts, and comparisons before drawing a conclusion.\n\n"
        "## Community contributors\n\n"
        f"This checkpoint incorporates `{accepted_assignments}` accepted work "
        f"units covering `{accepted_tokens}` loss-bearing tokens. "
        f"`{public_count}` contributing participant(s) chose named or "
        f"pseudonymous acknowledgment and `{anonymous_count}` chose anonymous "
        "credit.\n\n"
        f"{auxiliary_credit_text}\n\n"
        "[View the complete contributor acknowledgments](./CONTRIBUTORS.md). "
        "Accepted direct-training counts, optional worker-reported time, and "
        "approved direct-worker hardware classes are frozen in "
        "`attribution-snapshot.json`. Approved auxiliary work, time, hardware, "
        "and evidence identities are separate in "
        "`auxiliary-contribution-snapshot.json`.\n\n"
        "## Load and generate\n\n"
        "Install OrcaColony from the exact source revision below, then run:\n\n"
        "```bash\n"
        "python -m orcacolony.huggingface generate \\\n"
        "  --model . \\\n"
        '  --prompt "Once upon a time"\n'
        "```\n\n"
        "The package uses OrcaColony's custom `volunteer_decoder_v1` loader; it is "
        "not represented as a stock Transformers architecture.\n\n"
        "## Evidence and limitations\n\n"
        "- `campaign.json` freezes the training contract.\n"
        "- `evaluations.json` contains repeated validation evidence.\n"
        "- `language-model-final-holdout-evaluation.json`, when present, is a "
        "post-selection language-loss diagnostic.\n"
        "- `campaign-evaluation-evidence.json` and "
        "`campaign-evaluation-summary.json`, when present, preserve the "
        "campaign-owner-defined evaluator record, measurements, comparisons, "
        "findings, limitations, and reproduction command.\n"
        "- `promotion-evidence.json`, when present, is a legacy v1 campaign "
        "record retained for historical reproducibility.\n"
        "- `attribution-snapshot.json` and `CONTRIBUTORS.md` preserve release-time credit.\n"
        "- `auxiliary-contribution-snapshot.json` and "
        "`auxiliary-contribution-artifacts/`, when present, preserve separately "
        "reviewed auxiliary work and evidence without presenting it as accepted "
        "training.\n"
        "- `checkpoint-state.json` and `optimizer.safetensors` preserve the selected "
        "restart trajectory, even though generation needs only the weights.\n"
        "- `orcacolony-release.json` and `release-SHA256SUMS` bind this package to "
        "the operational release.\n\n"
        "## Intended use and limitations\n\n"
        "This is a research artifact for reproducing, inspecting, and testing the "
        "declared campaign. Review the frozen evaluation evidence before relying "
        "on it for any downstream use.\n\n"
        f"{limitation_lines}\n"
        f"Source: [{source_repository}]({source_repository}) at `{source_revision}`.\n"
    )


def _dataset_card(
    *,
    campaign_id: str,
    dataset_repo_id: str,
    dataset_license: str,
    source_repository: str,
    source_revision: str,
    dataset_manifest: Mapping[str, object],
    attribution: Mapping[str, object],
    auxiliary_contributions: Mapping[str, object],
) -> str:
    source = dataset_manifest.get("source")
    packing = dataset_manifest.get("packing")
    if not isinstance(source, Mapping) or not isinstance(packing, Mapping):
        raise ValueError("release dataset card metadata is invalid")
    accepted_assignments, accepted_tokens, public_count, anonymous_count = (
        _card_attribution(attribution)
    )
    (
        auxiliary_status,
        auxiliary_contributor_count,
        auxiliary_contribution_count,
        _,
        _,
    ) = _card_auxiliary_contributions(auxiliary_contributions)
    auxiliary_text = (
        f"The owner-reviewed auxiliary record contains "
        f"`{auxiliary_contribution_count}` contribution(s) from "
        f"`{auxiliary_contributor_count}` contributor(s)."
        if auxiliary_status == "owner_reviewed"
        else (
            "No owner-reviewed auxiliary contribution record was supplied; "
            "this private review package is not ready for public publication."
        )
    )
    return (
        "---\n"
        f"license: {dataset_license}\n"
        "task_categories:\n"
        "- text-generation\n"
        "pretty_name: OrcaColony frozen campaign dataset\n"
        "tags:\n"
        "- orcacolony\n"
        "- packed-token-dataset\n"
        "---\n\n"
        f"# Dataset for {campaign_id}\n\n"
        f"Repository target: `{dataset_repo_id}`.\n\n"
        "This repository contains the exact tokenizer and packed shifted-token "
        "artifacts admitted by the campaign. It is intended for reproducibility, "
        "not as a replacement for the canonical upstream raw dataset.\n\n"
        "Read `DATASET-NOTICE.md`, `THIRD_PARTY_DATA.md`, and `manifest.json` before "
        "reuse. Those files identify the upstream source, exact revision, selection, "
        "transformations, and byte/tensor hashes.\n\n"
        "When the campaign supplied usage-evaluation evidence, "
        "`campaign-evaluation-evidence.json`, "
        "`campaign-evaluation-summary.json`, and "
        "`campaign-evaluation-artifacts/` preserve the declared measurements "
        "and review files alongside the data.\n\n"
        "## Frozen source and packing\n\n"
        f"- Upstream dataset: `{source.get('dataset')}`.\n"
        f"- Upstream revision: `{source.get('revision')}`.\n"
        f"- Selection: `{source.get('selection')}`.\n"
        f"- Packed training sequences/tokens: "
        f"`{packing.get('train_sequences')}` / `{packing.get('train_tokens')}`.\n"
        f"- Packed validation sequences/tokens: "
        f"`{packing.get('validation_sequences')}` / "
        f"`{packing.get('validation_tokens')}`.\n"
        f"- Dataset license: `{dataset_license}`.\n\n"
        "## Community contributors\n\n"
        f"The linked training release contains `{accepted_assignments}` accepted "
        f"work units covering `{accepted_tokens}` loss-bearing tokens. "
        f"`{public_count}` contributor(s) chose public acknowledgment and "
        f"`{anonymous_count}` chose anonymous credit. "
        "[View the complete acknowledgments](./CONTRIBUTORS.md).\n\n"
        f"{auxiliary_text}\n\n"
        f"Packaging source: [{source_repository}]({source_repository}) at "
        f"`{source_revision}`.\n"
    )


def _package_file_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build_huggingface_packages(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    model_repo_id: str,
    dataset_repo_id: str,
    model_license: str,
    dataset_license: str,
    source_repository: str,
    source_revision: str,
    visibility: str = "private",
) -> dict[str, object]:
    """Build deterministic model and dataset repositories without network access."""

    model_repo_id = _validate_repo_id(model_repo_id, "model repository ID")
    dataset_repo_id = _validate_repo_id(dataset_repo_id, "dataset repository ID")
    model_license = _validate_license(model_license, "model license")
    dataset_license = _validate_license(dataset_license, "dataset license")
    source_repository = _validate_https_url(
        source_repository,
        "source repository",
    )
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source revision must be an exact lowercase Git commit")
    if visibility not in {"public", "private"}:
        raise ValueError("publication visibility must be public or private")

    release_root = Path(release_dir).resolve()
    release_manifest = verify_release_bundle(release_root)
    campaign = _load_mapping(release_root / "campaign.json", "campaign")
    campaign_lock = _load_mapping(
        release_root / "campaign-lock.json",
        "campaign lock",
    )
    dataset_manifest = _load_mapping(
        release_root / "dataset" / "manifest.json",
        "dataset manifest",
    )
    attribution = _load_mapping(
        release_root / "attribution-snapshot.json",
        "attribution snapshot",
    )
    auxiliary_contributions = _load_mapping(
        release_root / "auxiliary-contribution-snapshot.json",
        "auxiliary contribution snapshot",
    )
    validate_auxiliary_contribution_snapshot(auxiliary_contributions)
    verify_public_auxiliary_snapshot_artifacts(
        auxiliary_contributions,
        release_root / "auxiliary-contribution-artifacts",
    )
    auxiliary_status, _, _, _, _ = _card_auxiliary_contributions(
        auxiliary_contributions
    )
    dataset_source = dataset_manifest.get("source")
    if (
        not isinstance(dataset_source, Mapping)
        or dataset_source.get("license") != dataset_license
    ):
        raise ValueError(
            "requested dataset license differs from the frozen dataset source"
        )
    campaign_metadata = campaign.get("campaign")
    model_metadata = campaign.get("model")
    if not isinstance(campaign_metadata, Mapping) or not isinstance(
        model_metadata,
        Mapping,
    ):
        raise ValueError("release campaign model contract is invalid")
    campaign_id = campaign_metadata.get("id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("release campaign ID is invalid")
    if (
        auxiliary_contributions.get("campaign_id") != campaign_id
        or auxiliary_contributions.get("campaign_revision")
        != release_manifest.get("campaign_revision")
        or auxiliary_contributions.get("record_status")
        != release_manifest.get("auxiliary_contribution_record_status")
        or auxiliary_contributions.get("snapshot_sha256")
        != release_manifest.get("auxiliary_contribution_snapshot_sha256")
    ):
        raise ValueError(
            "auxiliary contribution snapshot differs from the release manifest"
        )
    if visibility == "public" and auxiliary_status != "owner_reviewed":
        raise ValueError(
            "public Hugging Face packaging requires an owner-reviewed "
            "auxiliary contribution record, including an explicit empty record "
            "when no auxiliary work occurred"
        )
    classification = release_manifest.get(
        "release_classification",
        "systems_evidence_only",
    )
    if classification not in {
        "systems_evidence_only",
        "campaign_checkpoint",
        "campaign_result",
        "capability_candidate",
        "capability_model",
    }:
        raise ValueError("release classification is invalid")
    if classification in {"campaign_checkpoint", "campaign_result"}:
        research = campaign.get("research")
        if (
            not isinstance(research, Mapping)
            or research.get("format") != "orcacolony_campaign_research_v2"
        ):
            raise ValueError(
                "campaign release classification requires a v2 research contract"
            )

    publication = campaign.get("publication")
    if isinstance(publication, Mapping):
        expected = {
            "model_repo_id": model_repo_id,
            "dataset_repo_id": dataset_repo_id,
            "model_license": model_license,
            "dataset_license": dataset_license,
        }
        if any(
            publication.get(field) != value
            for field, value in expected.items()
        ):
            raise ValueError(
                "requested Hugging Face targets differ from the campaign lock"
            )
        visibility_policy = publication.get("visibility_policy")
        permitted_visibility = {
            "private": {"private"},
            "public": {"public"},
            "private_review_then_public": {"private", "public"},
        }.get(visibility_policy)
        if (
            permitted_visibility is None
            or visibility not in permitted_visibility
        ):
            raise ValueError(
                "requested Hugging Face visibility differs from the campaign policy"
            )

    required_release_files = (
        "campaign.json",
        "campaign-lock.json",
        "evaluations.json",
        "public-ledger.json",
        "attribution-snapshot.json",
        "auxiliary-contribution-snapshot.json",
        "CONTRIBUTORS.md",
        "LICENSE",
        "THIRD_PARTY_DATA.md",
        "SHA256SUMS",
        "checkpoint/state.json",
        "checkpoint/optimizer.safetensors",
        "dataset/manifest.json",
        "dataset/tokenizer.json",
        "dataset/train.safetensors",
        "dataset/validation.safetensors",
        "dataset/DATASET-NOTICE.md",
    )
    for name in required_release_files:
        if not (release_root / name).is_file():
            raise ValueError(f"release is missing Hugging Face input: {name}")

    output = Path(output_dir).resolve()
    if output.is_relative_to(release_root):
        raise ValueError(
            "Hugging Face package output may not be inside the release"
        )
    if output.exists():
        raise ValueError(f"Hugging Face package output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    model_root = temporary / "model"
    dataset_root = temporary / "dataset"
    model_root.mkdir()
    dataset_root.mkdir()
    try:
        dense_model = release_root / "checkpoint" / "model.safetensors"
        adapter = release_root / "checkpoint" / "adapter.safetensors"
        if dense_model.is_file() and adapter.exists():
            raise ValueError("release contains both dense and adapter checkpoints")
        if dense_model.is_file():
            model_kind = "dense"
            _copy(dense_model, model_root / "model.safetensors")
        elif adapter.is_file():
            model_kind = "frozen-base-lora"
            for source_name, target_name in (
                ("checkpoint/base-model.safetensors", "base-model.safetensors"),
                ("checkpoint/adapter.safetensors", "adapter.safetensors"),
                ("lora.json", "lora.json"),
            ):
                _copy(release_root / source_name, model_root / target_name)
        else:
            raise ValueError("release does not contain a supported checkpoint")
        _copy(
            release_root / "checkpoint" / "state.json",
            model_root / "checkpoint-state.json",
        )
        _copy(
            release_root / "checkpoint" / "optimizer.safetensors",
            model_root / "optimizer.safetensors",
        )

        checkpoint = release_manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("release checkpoint manifest is invalid")
        release_checkpoint_sha256 = (
            checkpoint.get("resume_state_sha256")
            or checkpoint.get("model_sha256")
        )
        if (
            auxiliary_contributions.get("release_checkpoint_sha256")
            != release_checkpoint_sha256
            or auxiliary_contributions.get("release_checkpoint_step")
            != checkpoint.get("step")
        ):
            raise ValueError(
                "auxiliary contribution snapshot differs from the release checkpoint"
            )
        objective = campaign_metadata.get("objective")
        loss_mask = campaign_metadata.get("loss_mask")
        config: dict[str, object] = {
            "format": "orcacolony_huggingface_model_v1",
            "model_type": "orcacolony_volunteer_decoder",
            "architectures": ["VolunteerDecoder"],
            "library_name": "orcacolony",
            "campaign_id": campaign_id,
            "objective": objective,
            "loss_mask": loss_mask,
            "model_kind": model_kind,
            "model": dict(model_metadata),
            "checkpoint": dict(checkpoint),
            "dataset_repo_id": dataset_repo_id,
            "source": {
                "repository": source_repository,
                "revision": source_revision,
            },
        }
        _write_json(model_root / "config.json", config)
        for source_name, target_name in (
            ("campaign.json", "campaign.json"),
            ("campaign-lock.json", "campaign-lock.json"),
            ("evaluations.json", "evaluations.json"),
            ("public-ledger.json", "public-ledger.json"),
            ("attribution-snapshot.json", "attribution-snapshot.json"),
            (
                "auxiliary-contribution-snapshot.json",
                "auxiliary-contribution-snapshot.json",
            ),
            ("CONTRIBUTORS.md", "CONTRIBUTORS.md"),
            ("LICENSE", "ORCACOLONY-SOFTWARE-LICENSE"),
            ("THIRD_PARTY_DATA.md", "THIRD_PARTY_DATA.md"),
            ("release-manifest.json", "orcacolony-release.json"),
            ("SHA256SUMS", "release-SHA256SUMS"),
            ("dataset/tokenizer.json", "tokenizer.json"),
            ("dataset/manifest.json", "dataset-manifest.json"),
        ):
            _copy(release_root / source_name, model_root / target_name)
        (model_root / "MODEL-LICENSE.md").write_text(
            "# Model weights license\n\n"
            f"Hugging Face license identifier: `{model_license}`.\n\n"
            "This identifier applies to the released model weights. "
            "`ORCACOLONY-SOFTWARE-LICENSE` records the source framework's "
            "software license and does not replace or broaden the model-weights "
            "terms.\n",
            encoding="utf-8",
            newline="\n",
        )
        final_holdout = (
            release_root / "language-model-final-holdout-evaluation.json"
        )
        if final_holdout.is_file():
            _copy(final_holdout, model_root / final_holdout.name)
        elif classification == "capability_model":
            raise ValueError(
                "capability model release is missing final-holdout evidence"
            )
        promotion_evidence = release_root / "promotion-evidence.json"
        promotion_payload: Mapping[str, object] | None = None
        if promotion_evidence.is_file():
            promotion_payload = _load_mapping(
                promotion_evidence,
                "promotion evidence",
            )
            _copy(promotion_evidence, model_root / promotion_evidence.name)
        elif classification == "capability_model":
            raise ValueError(
                "capability model release is missing promotion evidence"
            )
        campaign_evaluation_path = (
            release_root / "campaign-evaluation-summary.json"
        )
        campaign_evidence_path = (
            release_root / "campaign-evaluation-evidence.json"
        )
        campaign_evaluation_payload: Mapping[str, object] | None = None
        if campaign_evaluation_path.is_file() and campaign_evidence_path.is_file():
            if classification != "campaign_result":
                raise ValueError(
                    "campaign evaluation evidence requires campaign-result "
                    "classification"
                )
            campaign_evaluation_payload = _load_mapping(
                campaign_evaluation_path,
                "campaign evaluation summary",
            )
            _copy(
                campaign_evaluation_path,
                model_root / campaign_evaluation_path.name,
            )
            _copy(
                campaign_evidence_path,
                model_root / campaign_evidence_path.name,
            )
            _copy(
                campaign_evaluation_path,
                dataset_root / campaign_evaluation_path.name,
            )
            _copy(
                campaign_evidence_path,
                dataset_root / campaign_evidence_path.name,
            )
            campaign_artifact_root = (
                release_root / "campaign-evaluation-artifacts"
            )
            if campaign_artifact_root.exists():
                for relative, source in _regular_file_map(
                    campaign_artifact_root,
                    "campaign evaluation artifacts",
                ).items():
                    _copy(
                        source,
                        model_root / "campaign-evaluation-artifacts" / relative,
                    )
                    _copy(
                        source,
                        dataset_root / "campaign-evaluation-artifacts" / relative,
                    )
        elif campaign_evaluation_path.exists() or campaign_evidence_path.exists():
            raise ValueError("campaign evaluation release files are incomplete")
        elif classification == "campaign_result":
            raise ValueError(
                "campaign result release is missing campaign evaluation evidence"
            )
        auxiliary_artifact_root = (
            release_root / "auxiliary-contribution-artifacts"
        )
        if auxiliary_artifact_root.exists():
            for relative, source in _regular_file_map(
                auxiliary_artifact_root,
                "auxiliary contribution artifacts",
            ).items():
                _copy(
                    source,
                    model_root
                    / "auxiliary-contribution-artifacts"
                    / relative,
                )
                _copy(
                    source,
                    dataset_root
                    / "auxiliary-contribution-artifacts"
                    / relative,
                )
        (model_root / "README.md").write_text(
            _model_card(
                campaign_id=campaign_id,
                model_repo_id=model_repo_id,
                dataset_repo_id=dataset_repo_id,
                model_license=model_license,
                dataset_license=dataset_license,
                release_classification=str(classification),
                source_repository=source_repository,
                source_revision=source_revision,
                has_language_model_holdout=final_holdout.is_file(),
                campaign=campaign,
                campaign_lock=campaign_lock,
                dataset_manifest=dataset_manifest,
                checkpoint=checkpoint,
                attribution=attribution,
                auxiliary_contributions=auxiliary_contributions,
                campaign_evaluation=campaign_evaluation_payload,
                promotion_evidence=promotion_payload,
            ),
            encoding="utf-8",
            newline="\n",
        )

        for name in (
            "manifest.json",
            "tokenizer.json",
            "train.safetensors",
            "validation.safetensors",
            "DATASET-NOTICE.md",
        ):
            _copy(release_root / "dataset" / name, dataset_root / name)
        for source_name, target_name in (
            ("THIRD_PARTY_DATA.md", "THIRD_PARTY_DATA.md"),
            ("release-manifest.json", "orcacolony-release.json"),
            ("SHA256SUMS", "release-SHA256SUMS"),
            ("attribution-snapshot.json", "attribution-snapshot.json"),
            (
                "auxiliary-contribution-snapshot.json",
                "auxiliary-contribution-snapshot.json",
            ),
            ("CONTRIBUTORS.md", "CONTRIBUTORS.md"),
        ):
            _copy(release_root / source_name, dataset_root / target_name)
        (dataset_root / "DATASET-LICENSE.md").write_text(
            "# Dataset license\n\n"
            f"Hugging Face license identifier: `{dataset_license}`.\n\n"
            "The identifier matches the frozen upstream license in "
            "`manifest.json`. Review `DATASET-NOTICE.md` and "
            "`THIRD_PARTY_DATA.md` for source and transformation details.\n",
            encoding="utf-8",
            newline="\n",
        )
        (dataset_root / "README.md").write_text(
            _dataset_card(
                campaign_id=campaign_id,
                dataset_repo_id=dataset_repo_id,
                dataset_license=dataset_license,
                source_repository=source_repository,
                source_revision=source_revision,
                dataset_manifest=dataset_manifest,
                attribution=attribution,
                auxiliary_contributions=auxiliary_contributions,
            ),
            encoding="utf-8",
            newline="\n",
        )

        model_files = _package_file_map(model_root)
        dataset_files = _package_file_map(dataset_root)
        manifest: dict[str, object] = {
            "format": "orcacolony_huggingface_publication_v1",
            "campaign_id": campaign_id,
            "release_manifest_sha256": _sha256_file(
                release_root / "release-manifest.json"
            ),
            "release_classification": classification,
            "auxiliary_contribution_record_status": auxiliary_status,
            "auxiliary_contribution_snapshot_sha256": (
                auxiliary_contributions["snapshot_sha256"]
            ),
            "visibility": visibility,
            "targets": {
                "model": {
                    "repo_id": model_repo_id,
                    "repo_type": "model",
                    "license": model_license,
                },
                "dataset": {
                    "repo_id": dataset_repo_id,
                    "repo_type": "dataset",
                    "license": dataset_license,
                },
            },
            "source": {
                "repository": source_repository,
                "revision": source_revision,
            },
            "files": {
                "model": model_files,
                "dataset": dataset_files,
            },
        }
        _write_json(temporary / "publication-manifest.json", manifest)
        checksums = {
            **{
                f"model/{name}": digest
                for name, digest in model_files.items()
            },
            **{
                f"dataset/{name}": digest
                for name, digest in dataset_files.items()
            },
            "publication-manifest.json": _sha256_file(
                temporary / "publication-manifest.json"
            ),
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{checksums[name]}  {name}\n"
                for name in sorted(checksums)
            ),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_huggingface_packages(package_dir: str | Path) -> Mapping[str, object]:
    root = Path(package_dir).resolve()
    manifest = _load_mapping(
        root / "publication-manifest.json",
        "publication manifest",
    )
    if manifest.get("format") != "orcacolony_huggingface_publication_v1":
        raise ValueError("unsupported Hugging Face publication package")
    if manifest.get("visibility") not in {"public", "private"}:
        raise ValueError("publication visibility is invalid")
    auxiliary_status = manifest.get(
        "auxiliary_contribution_record_status"
    )
    auxiliary_snapshot_sha256 = manifest.get(
        "auxiliary_contribution_snapshot_sha256"
    )
    if (
        auxiliary_status not in {"not_supplied", "owner_reviewed"}
        or not isinstance(auxiliary_snapshot_sha256, str)
        or len(auxiliary_snapshot_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in auxiliary_snapshot_sha256
        )
    ):
        raise ValueError(
            "publication auxiliary contribution identity is invalid"
        )
    if (
        manifest.get("visibility") == "public"
        and auxiliary_status != "owner_reviewed"
    ):
        raise ValueError(
            "public publication package lacks an owner-reviewed "
            "auxiliary contribution record"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("publication file map is missing")
    actual_files = _regular_file_map(root, "publication package")
    expected_root_files = {
        "publication-manifest.json",
        "SHA256SUMS",
    }
    checksum_entries: dict[str, str] = {}
    for repo_type in ("model", "dataset"):
        repo_files = files.get(repo_type)
        if not isinstance(repo_files, Mapping) or not repo_files:
            raise ValueError(f"publication {repo_type} file map is missing")
        for name, expected in repo_files.items():
            path = _safe_relative_path(name, f"{repo_type} package file")
            package_path = f"{repo_type}/{path.as_posix()}"
            expected_root_files.add(package_path)
            candidate = actual_files.get(package_path)
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected
                )
                or candidate is None
                or not candidate.is_file()
                or candidate.is_symlink()
                or _sha256_file(candidate) != expected
            ):
                raise ValueError(
                    f"publication {repo_type} file differs: {path.as_posix()}"
                )
            checksum_entries[package_path] = expected
    if set(actual_files) != expected_root_files:
        raise ValueError(
            "publication package contains an unmanifested or missing file"
        )
    for repository in ("model", "dataset"):
        auxiliary = _load_mapping(
            root
            / repository
            / "auxiliary-contribution-snapshot.json",
            f"{repository} auxiliary contribution snapshot",
        )
        validate_auxiliary_contribution_snapshot(auxiliary)
        if (
            auxiliary.get("record_status") != auxiliary_status
            or auxiliary.get("snapshot_sha256")
            != auxiliary_snapshot_sha256
        ):
            raise ValueError(
                "publication auxiliary contribution identity differs"
            )
    checksum_entries["publication-manifest.json"] = _sha256_file(
        root / "publication-manifest.json"
    )
    expected_checksums = "".join(
        f"{checksum_entries[name]}  {name}\n"
        for name in sorted(checksum_entries)
    )
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("publication SHA256SUMS differs from its manifest")
    targets = manifest.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != {"model", "dataset"}:
        raise ValueError("publication targets are invalid")
    for name, repo_type in (("model", "model"), ("dataset", "dataset")):
        target = targets.get(name)
        if not isinstance(target, Mapping) or set(target) != {
            "repo_id",
            "repo_type",
            "license",
        }:
            raise ValueError(f"publication {name} target is invalid")
        _validate_repo_id(str(target.get("repo_id")), f"{name} repo ID")
        _validate_license(str(target.get("license")), f"{name} license")
        if target.get("repo_type") != repo_type:
            raise ValueError(f"publication {name} repository type differs")
    return manifest


def load_model_package(
    model_dir: str | Path,
) -> tuple[object, object]:
    from safetensors.torch import load_file as load_safetensors_file
    from tokenizers import Tokenizer

    from .peft import LoRAConfig, _apply_lora_to_model, load_adapter_state
    from .reference import (
        ModelConfig,
        ObjectiveConfig,
        VolunteerDecoder,
        tensor_sha256,
    )

    root = Path(model_dir).resolve()
    _regular_file_map(root, "model package")
    config = _load_mapping(root / "config.json", "model package config")
    if config.get("format") != "orcacolony_huggingface_model_v1":
        raise ValueError("unsupported OrcaColony model package")
    raw_model = config.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("model package architecture is invalid")
    objective_name = config.get("objective")
    loss_mask = config.get("loss_mask")
    if objective_name != "causal_lm" or loss_mask != "all_target_tokens":
        raise ValueError("model package objective is unsupported")
    model_config = ModelConfig(**raw_model)  # type: ignore[arg-type]
    objective = ObjectiveConfig(name=objective_name, loss_mask=loss_mask)
    model_kind = config.get("model_kind")
    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("model package checkpoint identity is invalid")
    if model_kind == "dense":
        model_path = root / "model.safetensors"
        if _sha256_file(model_path) != checkpoint.get("model_sha256"):
            raise ValueError("dense model package digest differs")
        model = VolunteerDecoder(model_config, objective)
        state = load_safetensors_file(str(model_path))
        model.load_state_dict(state, strict=True)
        expected_parameter_count = model_config.parameters
    elif model_kind == "frozen-base-lora":
        lora_payload = _load_mapping(root / "lora.json", "LoRA package config")
        if lora_payload.get("format") != "orcacolony_release_lora_v1":
            raise ValueError("LoRA package format is invalid")
        raw_lora = lora_payload.get("config")
        if not isinstance(raw_lora, Mapping):
            raise ValueError("LoRA package config is invalid")
        lora_config = LoRAConfig(
            format=str(raw_lora["format"]),
            base_model_sha256=str(raw_lora["base_model_sha256"]),
            rank=int(raw_lora["rank"]),
            alpha=float(raw_lora["alpha"]),
            dropout=float(raw_lora["dropout"]),
            adapter_seed=int(raw_lora["adapter_seed"]),
            initialization_std=float(raw_lora["initialization_std"]),
            targets=tuple(raw_lora["targets"]),  # type: ignore[arg-type]
        )
        campaign_payload = _load_mapping(
            root / "campaign.json",
            "packaged campaign",
        )
        from .reference import campaign_from_mapping

        campaign = campaign_from_mapping(campaign_payload)
        if campaign.model != model_config or campaign.objective != objective:
            raise ValueError("LoRA package campaign differs from model config")
        base_path = root / "base-model.safetensors"
        base_state = load_safetensors_file(str(base_path))
        if tensor_sha256(base_state) != lora_config.base_model_sha256:
            raise ValueError("LoRA package base-model digest differs")
        base_model = VolunteerDecoder(model_config, objective)
        base_model.load_state_dict(base_state, strict=True)
        model = _apply_lora_to_model(base_model, lora_config)
        adapter_state = load_safetensors_file(str(root / "adapter.safetensors"))
        if tensor_sha256(adapter_state) != checkpoint.get("adapter_sha256"):
            raise ValueError("LoRA package adapter digest differs")
        load_adapter_state(model, adapter_state)
        expected_parameter_count = model_config.parameters + sum(
            tensor.numel() for tensor in adapter_state.values()
        )
    else:
        raise ValueError("model package kind is unsupported")
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != expected_parameter_count:
        raise ValueError("loaded model parameter count differs")
    model.eval()
    dataset_manifest = _load_mapping(
        root / "dataset-manifest.json",
        "packaged dataset manifest",
    )
    dataset_files = dataset_manifest.get("files")
    if (
        not isinstance(dataset_files, Mapping)
        or dataset_files.get("tokenizer.json")
        != _sha256_file(root / "tokenizer.json")
    ):
        raise ValueError("packaged tokenizer digest differs")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.get_vocab_size() > model_config.vocabulary_size:
        raise ValueError("packaged tokenizer vocabulary exceeds the model")
    return model, tokenizer


def generate(
    model_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
) -> str:
    import torch

    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be positive")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature < 0
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("sampling parameters are invalid")
    model, tokenizer = load_model_package(model_dir)
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = list(encoded.ids)
    if not token_ids:
        bos_id = tokenizer.token_to_id("<bos>")
        if bos_id is None:
            raise ValueError("empty prompt requires a <bos> token")
        token_ids = [bos_id]
    eos_id = tokenizer.token_to_id("<eos>")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    context_length = model.config.context_length
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[-context_length:]
            logits = model(torch.tensor([context], dtype=torch.long))[0, -1]
            if temperature == 0:
                next_token = int(torch.argmax(logits).item())
            else:
                scaled = logits / temperature
                if top_k:
                    k = min(top_k, scaled.numel())
                    values, indices = torch.topk(scaled, k)
                    sampled = torch.multinomial(
                        torch.softmax(values, dim=-1),
                        1,
                        generator=generator,
                    )
                    next_token = int(indices[sampled].item())
                else:
                    next_token = int(
                        torch.multinomial(
                            torch.softmax(scaled, dim=-1),
                            1,
                            generator=generator,
                        ).item()
                    )
            token_ids.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def publish_huggingface_packages(
    package_dir: str | Path,
    *,
    commit_message: str,
) -> dict[str, object]:
    """Create/upload both repositories using the locally authenticated HF user."""

    if not isinstance(commit_message, str) or not commit_message.strip():
        raise ValueError("Hugging Face commit message must be non-empty")
    manifest = verify_huggingface_packages(package_dir)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "publishing requires huggingface_hub; install it and run `hf auth login`"
        ) from exc
    targets = manifest.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("publication targets are invalid")
    api = HfApi()
    identity = api.whoami()
    if not isinstance(identity, Mapping) or not isinstance(
        identity.get("name"),
        str,
    ):
        raise RuntimeError("Hugging Face authentication identity is unavailable")
    private = manifest.get("visibility") == "private"
    root = Path(package_dir).resolve()
    plans: list[dict[str, object]] = []
    for name, repo_type in (("dataset", "dataset"), ("model", "model")):
        target = targets.get(name)
        if not isinstance(target, Mapping):
            raise ValueError(f"publication {name} target is invalid")
        repo_id = _validate_repo_id(str(target.get("repo_id")), f"{name} repo ID")
        exists = bool(api.repo_exists(repo_id=repo_id, repo_type=repo_type))
        if exists:
            remote = api.repo_info(repo_id=repo_id, repo_type=repo_type)
            remote_private = getattr(remote, "private", None)
            if type(remote_private) is not bool or remote_private != private:
                raise RuntimeError(
                    f"Hugging Face {repo_id} visibility differs from the "
                    "verified publication package"
                )
            siblings = getattr(remote, "siblings", None)
            if not isinstance(siblings, list):
                raise RuntimeError(
                    f"Hugging Face {repo_id} file inventory is unavailable"
                )
            remote_files = {
                str(getattr(sibling, "rfilename"))
                for sibling in siblings
                if isinstance(getattr(sibling, "rfilename", None), str)
            }
            local_files = set(
                _regular_file_map(root / name, f"{name} upload package")
            )
            unexpected = sorted(
                remote_files - local_files - {".gitattributes"}
            )
            if unexpected:
                raise RuntimeError(
                    f"Hugging Face {repo_id} contains files absent from the "
                    "verified package: "
                    + ", ".join(unexpected)
                )
        plans.append(
            {
                "name": name,
                "repo_type": repo_type,
                "repo_id": repo_id,
                "exists": exists,
            }
        )

    for plan in plans:
        if not plan["exists"]:
            api.create_repo(
                repo_id=str(plan["repo_id"]),
                repo_type=str(plan["repo_type"]),
                private=private,
                exist_ok=False,
            )
        remote = api.repo_info(
            repo_id=str(plan["repo_id"]),
            repo_type=str(plan["repo_type"]),
        )
        remote_private = getattr(remote, "private", None)
        if type(remote_private) is not bool or remote_private != private:
            raise RuntimeError(
                f"Hugging Face {plan['repo_id']} visibility verification failed"
            )

    results: dict[str, object] = {}
    for plan in plans:
        name = str(plan["name"])
        repo_type = str(plan["repo_type"])
        repo_id = str(plan["repo_id"])
        info = api.upload_folder(
            folder_path=str(root / name),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        results[name] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_url": str(info),
            "commit_sha": getattr(info, "oid", None),
        }
    return {
        "format": "orcacolony_huggingface_publish_result_v1",
        "campaign_id": manifest["campaign_id"],
        "authenticated_user": identity["name"],
        "visibility": manifest["visibility"],
        "repositories": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, verify, use, or explicitly publish OrcaColony Hub packages"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--release", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--model-repo-id", required=True)
    build.add_argument("--dataset-repo-id", required=True)
    build.add_argument("--model-license", required=True)
    build.add_argument("--dataset-license", required=True)
    build.add_argument(
        "--source-repository",
        default="https://github.com/zeidalidiez/OrcaColony",
    )
    build.add_argument("--source-revision", required=True)
    build.add_argument(
        "--visibility",
        choices=("public", "private"),
        required=True,
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)

    generation = subparsers.add_parser("generate")
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-k", type=int, default=0)
    generation.add_argument("--seed", type=int, default=0)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--package", type=Path, required=True)
    publish.add_argument("--commit-message", required=True)
    publish.add_argument(
        "--result",
        type=Path,
        required=True,
        help="new local JSON record for the authenticated user and Hub commits",
    )
    publish.add_argument(
        "--confirm-upload",
        action="store_true",
        help="required acknowledgement that this mutates Hugging Face repositories",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_huggingface_packages(
            args.release,
            args.output,
            model_repo_id=args.model_repo_id,
            dataset_repo_id=args.dataset_repo_id,
            model_license=args.model_license,
            dataset_license=args.dataset_license,
            source_repository=args.source_repository,
            source_revision=args.source_revision,
            visibility=args.visibility,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "verify":
        result = verify_huggingface_packages(args.package)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "generate":
        print(
            generate(
                args.model,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
            )
        )
    elif args.command == "publish":
        if not args.confirm_upload:
            raise ValueError("publish requires --confirm-upload")
        result_path = args.result.resolve()
        package_path = args.package.resolve()
        if result_path.is_relative_to(package_path):
            raise ValueError("publish result may not alter the verified package")
        if result_path.exists():
            raise ValueError(f"publish result already exists: {result_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = publish_huggingface_packages(
            args.package,
            commit_message=args.commit_message,
        )
        _write_new_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        raise AssertionError("unreachable command")


if __name__ == "__main__":
    main()
