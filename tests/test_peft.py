from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from orcacolony import peft
from orcacolony.peft import (
    LoRAConfig,
    adapter_named_parameters,
    apply_adapter_gradient_step,
    build_lora_model,
    compute_adapter_gradients,
    create_adapter_optimizer,
    export_lora_fixture,
    load_lora_manifest,
)
from orcacolony.reference import (
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"


def _lora_config() -> LoRAConfig:
    campaign = load_campaign(CONFIG)
    base = build_model(campaign)
    return LoRAConfig(
        format="orcacolony_lora_v1",
        base_model_sha256=tensor_sha256(base.state_dict()),
        rank=4,
        alpha=8.0,
        dropout=0.0,
        adapter_seed=20260725,
        initialization_std=0.01,
        targets=tuple(
            f"blocks.{index}.attention.qkv" for index in range(campaign.model.layers)
        ),
    )


def _base_parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    adapter_names = set(adapter_named_parameters(model))
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name not in adapter_names
    }


def test_lora_fixture_freezes_the_base_and_exports_complete_gradients() -> None:
    campaign = load_campaign(CONFIG)
    config = _lora_config()
    dense = build_model(campaign)
    model = build_lora_model(campaign, config)
    inputs, targets = fixture_batch(campaign)

    assert torch.equal(model(inputs), dense(inputs))
    adapters = adapter_named_parameters(model)
    assert list(adapters) == [
        name
        for layer in range(campaign.model.layers)
        for name in (
            f"blocks.{layer}.attention.qkv.lora_a",
            f"blocks.{layer}.attention.qkv.lora_b",
        )
    ]
    assert sum(parameter.numel() for parameter in adapters.values()) == 8_192
    assert all(parameter.requires_grad for parameter in adapters.values())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in adapters
    )

    first = compute_adapter_gradients(model, inputs, targets)
    repeated_model = build_lora_model(campaign, config)
    repeated = compute_adapter_gradients(repeated_model, inputs, targets)

    assert first.loss_sum == repeated.loss_sum
    assert first.loss_weight_sum == targets.numel()
    assert first.gradient_sha256 == repeated.gradient_sha256
    assert list(first.gradients) == list(adapters)
    assert all(torch.isfinite(gradient).all() for gradient in first.gradients.values())
    for layer in range(campaign.model.layers):
        assert first.gradients[f"blocks.{layer}.attention.qkv.lora_a"].shape == (
            4,
            128,
        )
        assert first.gradients[f"blocks.{layer}.attention.qkv.lora_b"].shape == (
            384,
            4,
        )


def test_adapter_gradient_application_matches_an_independent_mean_loss_step() -> None:
    campaign = load_campaign(CONFIG)
    config = _lora_config()
    inputs, targets = fixture_batch(campaign)
    worker = build_lora_model(campaign, config)
    coordinator = build_lora_model(campaign, config)
    reference = build_lora_model(campaign, config)
    coordinator_base_before = _base_parameter_snapshot(coordinator)
    reference_base_before = _base_parameter_snapshot(reference)

    submitted = compute_adapter_gradients(worker, inputs, targets)
    coordinator_optimizer = create_adapter_optimizer(coordinator, campaign.training)
    apply_adapter_gradient_step(
        coordinator,
        coordinator_optimizer,
        submitted.gradients,
        submitted.loss_weight_sum,
        campaign.training.max_gradient_norm,
    )

    reference_optimizer = create_adapter_optimizer(reference, campaign.training)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss_sum = F.cross_entropy(
        reference(inputs).reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="sum",
    )
    (reference_loss_sum / targets.numel()).backward()
    torch.nn.utils.clip_grad_norm_(
        list(adapter_named_parameters(reference).values()),
        campaign.training.max_gradient_norm,
    )
    reference_optimizer.step()

    for name, parameter in adapter_named_parameters(coordinator).items():
        assert torch.equal(parameter, adapter_named_parameters(reference)[name])
    for name, parameter in _base_parameter_snapshot(coordinator).items():
        assert torch.equal(parameter, coordinator_base_before[name])
    for name, parameter in _base_parameter_snapshot(reference).items():
        assert torch.equal(parameter, reference_base_before[name])


def test_lora_fixture_export_is_deterministic_and_self_describing(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)

    first = export_lora_fixture(loaded, tmp_path / "first")
    second = export_lora_fixture(loaded, tmp_path / "second")

    expected_files = (
        "base.safetensors",
        "adapter.safetensors",
        "batch.safetensors",
        "gradients.safetensors",
        "updated-adapter.safetensors",
        "fixture.json",
        "SHA256SUMS",
    )
    for filename in expected_files:
        assert (first.output_dir / filename).read_bytes() == (
            second.output_dir / filename
        ).read_bytes()
    checksum_bytes = (first.output_dir / "SHA256SUMS").read_bytes()
    assert b"\r" not in checksum_bytes
    for line in checksum_bytes.decode("utf-8").splitlines():
        expected_sha256, filename = line.split("  ", maxsplit=1)
        assert hashlib.sha256(
            (first.output_dir / filename).read_bytes()
        ).hexdigest() == expected_sha256
    manifest = json.loads(
        (first.output_dir / "fixture.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "orcacolony_lora_fixture_v1"
    assert manifest["base"]["model_sha256"] == loaded.config.base_model_sha256
    assert manifest["input_shape"] == [4, 128]
    assert len(manifest["input_ids"]) == 512
    assert len(manifest["target_ids"]) == 512
    assert manifest["model"] == {
        "context_length": 128,
        "d_ff": 512,
        "d_model": 128,
        "num_heads": 2,
        "num_layers": 4,
        "vocab_size": 4096,
    }
    assert manifest["adapter"]["tensor_count"] == 8
    assert manifest["adapter"]["value_count"] == 8_192
    assert manifest["gradient_contract"] == {
        "accumulation_dtype": "float32",
        "loss_reduction": "sum",
        "normalization_owner": "coordinator",
        "tensor_set": "complete_adapter_manifest",
    }
    assert manifest["one_step_update"]["matches_mean_loss_reference"] is True

    base = load_file(str(first.output_dir / "base.safetensors"))
    adapter = load_file(str(first.output_dir / "adapter.safetensors"))
    gradients = load_file(str(first.output_dir / "gradients.safetensors"))
    updated = load_file(str(first.output_dir / "updated-adapter.safetensors"))
    assert all("lora_" not in name for name in base)
    assert sorted(adapter) == manifest["adapter"]["tensor_order"]
    assert sorted(gradients) == sorted(adapter)
    assert sorted(updated) == sorted(adapter)
    assert any(not torch.equal(adapter[name], updated[name]) for name in adapter)


def test_lora_fixture_cli_exports_the_exact_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "cli-fixture"

    peft.main(
        [
            "export-fixture",
            "--campaign",
            str(CONFIG),
            "--lora",
            str(LORA_CONFIG),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["format"] == "orcacolony_lora_fixture_export_v1"
    assert summary["output_dir"] == str(output)
    assert summary["gradient_sha256"] == json.loads(
        (output / "fixture.json").read_text(encoding="utf-8")
    )["gradient"]["sha256"]


def test_lora_manifest_rejects_boolean_dropout(tmp_path: Path) -> None:
    campaign_copy = tmp_path / CONFIG.name
    campaign_copy.write_bytes(CONFIG.read_bytes())
    manifest_payload = json.loads(LORA_CONFIG.read_text(encoding="utf-8"))
    manifest_payload["adapter"]["dropout"] = False
    manifest_copy = tmp_path / LORA_CONFIG.name
    manifest_copy.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dropout must be a finite number"):
        load_lora_manifest(campaign_copy, manifest_copy)
