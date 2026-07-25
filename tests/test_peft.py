from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
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
    load_lora_checkpoint,
    load_lora_manifest,
    run_lora_training,
    save_lora_checkpoint,
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


def _saved_checkpoint(tmp_path: Path):
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    checkpoint = tmp_path / "checkpoint"
    run_lora_training(loaded, checkpoint, target_steps=1)
    return loaded, checkpoint


def _rewrite_optimizer(
    checkpoint: Path,
    tensors: dict[str, torch.Tensor],
) -> None:
    optimizer_path = checkpoint / "optimizer.safetensors"
    save_file(tensors, str(optimizer_path))
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["optimizer"]["sha256"] = hashlib.sha256(
        optimizer_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")


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


def test_adapter_state_validation_is_atomic() -> None:
    campaign = load_campaign(CONFIG)
    model = build_lora_model(campaign, _lora_config())
    adapters = adapter_named_parameters(model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in adapters.items()
    }
    replacement = {
        name: torch.full_like(parameter, float(index + 1))
        for index, (name, parameter) in enumerate(adapters.items())
    }
    malformed_name = list(adapters)[1]
    replacement[malformed_name] = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="adapter checkpoint shape differs"):
        peft.load_adapter_state(model, replacement)

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in adapters.items()
    )


def test_int8_frozen_linear_profile_reduces_resident_tensors_with_explicit_drift() -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    tokens = loaded.campaign.training.batch_size * loaded.campaign.model.context_length
    inputs = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        loaded.campaign.training.active_vocabulary_size
    ).reshape(
        loaded.campaign.training.batch_size,
        loaded.campaign.model.context_length,
    )
    targets = (inputs + 1).remainder(
        loaded.campaign.training.active_vocabulary_size
    )
    fp32_model = peft.build_lora_model(loaded.campaign, loaded.config)
    int8_model = peft.build_int8_lora_model(loaded.campaign, loaded.config)

    def resident_tensor_bytes(model: torch.nn.Module) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in [*model.parameters(), *model.buffers()]:
            pointer = tensor.untyped_storage().data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += tensor.untyped_storage().nbytes()
        return total

    quantized_linears = [
        module
        for module in int8_model.modules()
        if isinstance(module, peft.Int8FrozenLinear)
    ]
    assert peft.INT8_FROZEN_LINEAR_PROFILE == (
        "int8-per-output-symmetric-f32-dequant-v1"
    )
    assert len(quantized_linears) == 16
    with pytest.raises(ValueError, match="requires FP32 activations"):
        quantized_linears[0](
            torch.zeros(1, quantized_linears[0].in_features, dtype=torch.float16)
        )
    autocast_input = torch.zeros(
        1,
        quantized_linears[0].in_features,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = quantized_linears[0](autocast_input)
    assert autocast_output.dtype == torch.float32
    autocast_output.sum().backward()
    assert autocast_input.grad is not None
    assert autocast_input.grad.dtype == torch.float32
    assert resident_tensor_bytes(int8_model) < resident_tensor_bytes(fp32_model) * 0.6
    assert list(peft.adapter_named_parameters(int8_model)) == list(
        peft.adapter_named_parameters(fp32_model)
    )
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in int8_model.named_parameters()
    )

    fp32 = peft.compute_adapter_gradients(fp32_model, inputs, targets)
    int8 = peft.compute_adapter_gradients(int8_model, inputs, targets)
    reference = torch.cat(
        [fp32.gradients[name].reshape(-1).double() for name in fp32.gradients]
    )
    candidate = torch.cat(
        [int8.gradients[name].reshape(-1).double() for name in fp32.gradients]
    )
    relative_l2 = float(
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference)
    )
    cosine = float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0))
    assert cosine > 0.9998
    assert 0.015 < relative_l2 < 0.019
    assert abs(int8.loss_sum - fp32.loss_sum) / abs(fp32.loss_sum) < 1e-4
    assert relative_l2 > 1e-3  # Deliberately not the connected FP32 profile.


def test_streamed_fp32_frozen_linear_profile_is_exact_and_detects_mutation(
    tmp_path: Path,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    tokens = loaded.campaign.training.batch_size * loaded.campaign.model.context_length
    inputs = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        loaded.campaign.training.active_vocabulary_size
    ).reshape(
        loaded.campaign.training.batch_size,
        loaded.campaign.model.context_length,
    )
    targets = (inputs + 1).remainder(
        loaded.campaign.training.active_vocabulary_size
    )
    fp32_model = peft.build_lora_model(loaded.campaign, loaded.config)
    streamed_model = peft.build_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        tmp_path / "streamed-linears",
    )

    def resident_tensor_bytes(model: torch.nn.Module) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in [*model.parameters(), *model.buffers()]:
            pointer = tensor.untyped_storage().data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += tensor.untyped_storage().nbytes()
        return total

    streamed_linears = [
        module
        for module in streamed_model.modules()
        if isinstance(module, peft.StreamedFrozenLinear)
    ]
    assert peft.STREAMED_FP32_FROZEN_LINEAR_PROFILE == (
        "streamed-fp32-frozen-linear-v1"
    )
    assert len(streamed_linears) == 16
    assert resident_tensor_bytes(streamed_model) < resident_tensor_bytes(fp32_model) * 0.5

    fp32 = peft.compute_adapter_gradients(fp32_model, inputs, targets)
    streamed = peft.compute_adapter_gradients(streamed_model, inputs, targets)
    assert streamed.loss_sum == fp32.loss_sum
    assert all(
        torch.equal(streamed.gradients[name], tensor)
        for name, tensor in fp32.gradients.items()
    )
    assert sum(module.read_count for module in streamed_linears) == 31

    with pytest.raises(ValueError, match="requires CPU FP32 activations"):
        streamed_linears[0](
            torch.zeros(1, streamed_linears[0].in_features, dtype=torch.float16)
        )
    autocast_input = torch.zeros(
        1,
        streamed_linears[0].in_features,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = streamed_linears[0](autocast_input)
    assert autocast_output.dtype == torch.float32
    autocast_output.sum().backward()
    assert autocast_input.grad is not None
    assert autocast_input.grad.dtype == torch.float32

    first = streamed_linears[0]
    snapshot_weight, _ = first.load_tensors()
    snapshot_before_mutation = snapshot_weight.clone()
    with first.artifact_path.open("r+b") as stream:
        stream.seek(-1, 2)
        final_byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([final_byte[0] ^ 0x01]))
        stream.flush()
    assert torch.equal(snapshot_weight, snapshot_before_mutation)
    with pytest.raises(ValueError, match="streamed linear tensor digest mismatch"):
        first.load_tensors()


def test_direct_streamed_profile_builds_from_artifacts_without_full_fp32_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    base_model = build_model(loaded.campaign)
    base_path = tmp_path / "base-model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in base_model.state_dict().items()
        },
        base_path,
    )
    base_artifact_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    resident_model = peft.build_lora_model(loaded.campaign, loaded.config)
    adapter_state = {
        name: tensor.detach().clone()
        for name, tensor in peft.adapter_named_parameters(resident_model).items()
    }
    tokens = loaded.campaign.training.batch_size * loaded.campaign.model.context_length
    inputs = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        loaded.campaign.training.active_vocabulary_size
    ).reshape(
        loaded.campaign.training.batch_size,
        loaded.campaign.model.context_length,
    )
    targets = (inputs + 1).remainder(
        loaded.campaign.training.active_vocabulary_size
    )

    def forbidden_full_builder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct streamed construction materialized the FP32 builder")

    monkeypatch.setattr(peft, "build_lora_model", forbidden_full_builder)
    with pytest.raises(ValueError, match="base artifact SHA-256 mismatch"):
        peft.build_direct_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            base_path,
            "0" * 64,
            adapter_state,
        )
    direct_model = peft.build_direct_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        base_path,
        base_artifact_sha256,
        adapter_state,
    )

    direct_linears = [
        module
        for module in direct_model.modules()
        if isinstance(module, peft.DirectStreamedFrozenLinear)
    ]
    assert peft.DIRECT_STREAMED_FP32_PROFILE == "direct-streamed-fp32-v1"
    assert len(direct_linears) == 16
    assert all(parameter.device.type == "cpu" for parameter in direct_model.parameters())

    resident = peft.compute_adapter_gradients(resident_model, inputs, targets)
    direct = peft.compute_adapter_gradients(direct_model, inputs, targets)
    assert direct.loss_sum == resident.loss_sum
    assert all(
        torch.equal(direct.gradients[name], tensor)
        for name, tensor in resident.gradients.items()
    )
    assert sum(module.read_count for module in direct_linears) == 31

    short_inputs = inputs[:, :8]
    short_targets = targets[:, :8]
    resident_short = peft.compute_adapter_gradients(
        resident_model,
        short_inputs,
        short_targets,
    )
    direct_short = peft.compute_adapter_gradients(
        direct_model,
        short_inputs,
        short_targets,
    )
    assert direct_short.loss_sum == resident_short.loss_sum
    assert all(
        torch.equal(direct_short.gradients[name], tensor)
        for name, tensor in resident_short.gradients.items()
    )

    restarted = peft.build_direct_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        base_path,
        base_artifact_sha256,
        adapter_state,
    )
    restarted_result = peft.compute_adapter_gradients(restarted, inputs, targets)
    assert restarted_result.loss_sum == resident.loss_sum
    assert all(
        torch.equal(restarted_result.gradients[name], tensor)
        for name, tensor in resident.gradients.items()
    )

    first_direct = direct_linears[0]
    mutated_base = {
        name: tensor.clone()
        for name, tensor in load_file(base_path, device="cpu").items()
    }
    mutated_base[first_direct.weight_key][0, 0] += 1.0
    save_file(mutated_base, base_path)
    with pytest.raises(ValueError, match="direct streamed tensor digest mismatch"):
        first_direct.load_tensors()


def test_layer_bundle_builds_exact_model_without_linear_startup_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    base_model = build_model(loaded.campaign)
    base_path = tmp_path / "base-model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in base_model.state_dict().items()
        },
        base_path,
    )
    base_artifact_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    bundle = peft.export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        base_path,
        base_artifact_sha256,
        tmp_path / "base-layer-bundle",
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "orcacolony_base_layer_bundle_v1"
    assert manifest["base_model_sha256"] == loaded.config.base_model_sha256
    assert bundle.linear_count == 16

    resident_model = peft.build_lora_model(loaded.campaign, loaded.config)
    adapter_state = {
        name: tensor.detach().clone()
        for name, tensor in peft.adapter_named_parameters(resident_model).items()
    }

    opened_artifacts: list[Path] = []
    original_load = peft.load_safetensors_file

    def recording_load(path: str | Path, *args: object, **kwargs: object):
        opened_artifacts.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    def forbidden_full_builder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("layer-bundle construction materialized the FP32 builder")

    monkeypatch.setattr(peft, "load_safetensors_file", recording_load)
    monkeypatch.setattr(peft, "build_lora_model", forbidden_full_builder)
    model = peft.build_layer_bundle_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        bundle.output_dir,
        bundle.manifest_sha256,
        adapter_state,
    )
    linears = [
        module
        for module in model.modules()
        if isinstance(module, peft.LayerBundleStreamedFrozenLinear)
    ]
    assert peft.LAYER_BUNDLE_STREAMED_FP32_PROFILE == (
        "layer-bundle-streamed-fp32-v1"
    )
    assert len(linears) == 16
    assert not any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert sum(module.read_count for module in linears) == 0
    assert opened_artifacts == [(bundle.output_dir / "resident.safetensors").resolve()]
    assert all(
        module.causal_mask.shape
        == (
            loaded.campaign.model.context_length,
            loaded.campaign.model.context_length,
        )
        for module in model.modules()
        if isinstance(module, peft.CausalSelfAttention)
    )

    inputs, targets = fixture_batch(loaded.campaign)
    resident = None
    for sequence_length in (1, 8, loaded.campaign.model.context_length):
        reads_before = sum(module.read_count for module in linears)
        resident = peft.compute_adapter_gradients(
            resident_model,
            inputs[:, :sequence_length],
            targets[:, :sequence_length],
        )
        streamed = peft.compute_adapter_gradients(
            model,
            inputs[:, :sequence_length],
            targets[:, :sequence_length],
        )
        assert streamed.loss_sum == resident.loss_sum
        assert all(
            torch.equal(streamed.gradients[name], tensor)
            for name, tensor in resident.gradients.items()
        )
        assert sum(module.read_count for module in linears) - reads_before == 31
    assert resident is not None

    restarted = peft.build_layer_bundle_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        bundle.output_dir,
        bundle.manifest_sha256,
        adapter_state,
    )
    restarted_linears = [
        module
        for module in restarted.modules()
        if isinstance(module, peft.LayerBundleStreamedFrozenLinear)
    ]
    assert sum(module.read_count for module in restarted_linears) == 0
    restarted_result = peft.compute_adapter_gradients(restarted, inputs, targets)
    assert restarted_result.loss_sum == resident.loss_sum
    assert all(
        torch.equal(restarted_result.gradients[name], tensor)
        for name, tensor in resident.gradients.items()
    )


def test_layer_bundle_builds_int8_directly_without_resident_fp32_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    base_model = build_model(loaded.campaign)
    base_path = tmp_path / "base-model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in base_model.state_dict().items()
        },
        base_path,
    )
    bundle = peft.export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        base_path,
        hashlib.sha256(base_path.read_bytes()).hexdigest(),
        tmp_path / "base-layer-bundle",
    )
    converted = peft.build_int8_lora_model(loaded.campaign, loaded.config)
    adapter_state = {
        name: tensor.detach().clone()
        for name, tensor in peft.adapter_named_parameters(converted).items()
    }

    def forbidden_full_builder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct int8 construction materialized the FP32 builder")

    monkeypatch.setattr(peft, "build_lora_model", forbidden_full_builder)
    direct = peft.build_layer_bundle_int8_lora_model(
        loaded.campaign,
        loaded.config,
        bundle.output_dir,
        bundle.manifest_sha256,
        adapter_state,
    )
    converted_linears = {
        name: module
        for name, module in converted.named_modules()
        if isinstance(module, peft.Int8FrozenLinear)
    }
    direct_linears = {
        name: module
        for name, module in direct.named_modules()
        if isinstance(module, peft.LayerBundleInt8FrozenLinear)
    }
    assert peft.LAYER_BUNDLE_INT8_PROFILE == (
        "layer-bundle-int8-per-output-symmetric-f32-dequant-v1"
    )
    assert len(direct_linears) == 16
    assert set(direct_linears) == set(converted_linears)
    assert not any(isinstance(module, torch.nn.Linear) for module in direct.modules())
    assert all(module.artifact_open_count == 1 for module in direct_linears.values())
    for name, module in direct_linears.items():
        reference = converted_linears[name]
        assert torch.equal(module.qweight, reference.qweight)
        assert torch.equal(module.scales, reference.scales)
        if reference.bias is None:
            assert module.bias is None
        else:
            assert torch.equal(module.bias, reference.bias)

    def resident_tensor_bytes(model: torch.nn.Module) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in [*model.parameters(), *model.buffers()]:
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                total += storage.nbytes()
        return total

    assert resident_tensor_bytes(direct) == resident_tensor_bytes(converted)
    inputs, targets = fixture_batch(loaded.campaign)
    for sequence_length in (1, 8, loaded.campaign.model.context_length):
        expected = peft.compute_adapter_gradients(
            converted,
            inputs[:, :sequence_length],
            targets[:, :sequence_length],
        )
        actual = peft.compute_adapter_gradients(
            direct,
            inputs[:, :sequence_length],
            targets[:, :sequence_length],
        )
        assert actual.loss_sum == expected.loss_sum
        assert actual.gradient_sha256 == expected.gradient_sha256
        assert all(
            torch.equal(actual.gradients[name], tensor)
            for name, tensor in expected.gradients.items()
        )

    restarted = peft.build_layer_bundle_int8_lora_model(
        loaded.campaign,
        loaded.config,
        bundle.output_dir,
        bundle.manifest_sha256,
        adapter_state,
    )
    expected_restart = peft.compute_adapter_gradients(converted, inputs, targets)
    actual_restart = peft.compute_adapter_gradients(restarted, inputs, targets)
    assert actual_restart.loss_sum == expected_restart.loss_sum
    assert actual_restart.gradient_sha256 == expected_restart.gradient_sha256

    first_direct = next(iter(direct_linears.values()))
    qweight_snapshot = first_direct.qweight.clone()
    mutated = {
        name: tensor.clone()
        for name, tensor in load_file(first_direct.artifact_path).items()
    }
    mutated["weight"].add_(1.0)
    save_file(mutated, first_direct.artifact_path)
    assert torch.equal(first_direct.qweight, qweight_snapshot)
    with pytest.raises(ValueError, match="layer-bundle linear tensor digest mismatch"):
        peft.build_layer_bundle_int8_lora_model(
            loaded.campaign,
            loaded.config,
            bundle.output_dir,
            bundle.manifest_sha256,
            adapter_state,
        )


def test_layer_bundle_can_parse_lora_contract_without_resident_base_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_full_builder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("LoRA manifest parser materialized the FP32 base")

    monkeypatch.setattr(peft, "build_lora_model", forbidden_full_builder)
    loaded = peft.load_lora_manifest(
        CONFIG,
        LORA_CONFIG,
        verify_base_model=False,
    )
    assert loaded.config.base_model_sha256 == _lora_config().base_model_sha256


@pytest.mark.parametrize("verify_base_model", [False, True])
def test_lora_manifest_parses_the_authenticated_campaign_bytes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_base_model: bool,
) -> None:
    campaign_path = tmp_path / CONFIG.name
    manifest_path = tmp_path / LORA_CONFIG.name
    campaign_text = CONFIG.read_text(encoding="utf-8")
    campaign_path.write_text(campaign_text, encoding="utf-8")
    manifest_path.write_bytes(LORA_CONFIG.read_bytes())
    mutated_payload = json.loads(campaign_text)
    mutated_payload["model"]["gelu_approximation"] = "none"
    mutated_text = json.dumps(mutated_payload)
    original_read_text = Path.read_text
    campaign_reads = 0

    def raced_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal campaign_reads
        if path.resolve() == campaign_path.resolve():
            campaign_reads += 1
            if campaign_reads > 1:
                return mutated_text
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raced_read_text)
    loaded = peft.load_lora_manifest(
        campaign_path,
        manifest_path,
        verify_base_model=verify_base_model,
    )

    assert campaign_reads == 1
    assert loaded.campaign.model.gelu_approximation == "tanh"


def test_layer_bundle_rejects_unlisted_artifacts_and_manifest_rebinding(
    tmp_path: Path,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    base_model = build_model(loaded.campaign)
    base_path = tmp_path / "base-model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in base_model.state_dict().items()
        },
        base_path,
    )
    bundle = peft.export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        base_path,
        hashlib.sha256(base_path.read_bytes()).hexdigest(),
        tmp_path / "base-layer-bundle",
    )
    adapter_state = {
        name: tensor.detach().clone()
        for name, tensor in peft.adapter_named_parameters(
            peft.build_lora_model(loaded.campaign, loaded.config)
        ).items()
    }

    unlisted_path = bundle.output_dir / "linear-99999.safetensors"
    save_file({"weight": torch.zeros(1, dtype=torch.float32)}, unlisted_path)
    with pytest.raises(ValueError, match="artifact set differs"):
        peft.build_layer_bundle_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            bundle.output_dir,
            bundle.manifest_sha256,
            adapter_state,
        )
    unlisted_path.unlink()

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    manifest["base_model_sha256"] = "0" * 64
    rebound_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    bundle.manifest_path.write_bytes(rebound_bytes)
    with pytest.raises(ValueError, match="does not match base_model_sha256"):
        peft.build_layer_bundle_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            bundle.output_dir,
            hashlib.sha256(rebound_bytes).hexdigest(),
            adapter_state,
        )


def test_layer_bundle_export_is_deterministic_and_layer_snapshots_are_owned(
    tmp_path: Path,
) -> None:
    loaded = peft.load_lora_manifest(CONFIG, LORA_CONFIG)
    base_model = build_model(loaded.campaign)
    base_path = tmp_path / "base-model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in base_model.state_dict().items()
        },
        base_path,
    )
    base_artifact_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    first = peft.export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        base_path,
        base_artifact_sha256,
        tmp_path / "first-bundle",
    )
    second = peft.export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        base_path,
        base_artifact_sha256,
        tmp_path / "second-bundle",
    )
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.artifact_bytes == second.artifact_bytes
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.output_dir.iterdir()
    } == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.output_dir.iterdir()
    }

    resident_model = peft.build_lora_model(loaded.campaign, loaded.config)
    model = peft.build_layer_bundle_streamed_lora_model(
        loaded.campaign,
        loaded.config,
        first.output_dir,
        first.manifest_sha256,
        {
            name: tensor.detach().clone()
            for name, tensor in peft.adapter_named_parameters(resident_model).items()
        },
    )
    first_linear = next(
        module
        for module in model.modules()
        if isinstance(module, peft.LayerBundleStreamedFrozenLinear)
    )
    weight_snapshot, bias_snapshot = first_linear.load_tensors()
    original_weight = weight_snapshot.clone()
    original_bias = None if bias_snapshot is None else bias_snapshot.clone()
    repeated_weight, repeated_bias = first_linear.load_tensors()
    assert repeated_weight.untyped_storage().data_ptr() != (
        weight_snapshot.untyped_storage().data_ptr()
    )
    repeated_weight.add_(1.0)
    assert torch.equal(weight_snapshot, original_weight)
    if bias_snapshot is not None and repeated_bias is not None:
        assert repeated_bias.untyped_storage().data_ptr() != (
            bias_snapshot.untyped_storage().data_ptr()
        )

    sample = torch.randn(2, 3, first_linear.in_features, dtype=torch.float32)
    ordinary_input = sample.clone().requires_grad_(True)
    ordinary_output = first_linear(ordinary_input)
    ordinary_output.sum().backward()
    ordinary_gradient = ordinary_input.grad.detach().clone()
    autocast_input = sample.clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_output = first_linear(autocast_input)
    assert autocast_output.dtype == torch.float32
    autocast_output.sum().backward()
    assert autocast_input.grad is not None
    assert autocast_input.grad.dtype == torch.float32
    assert torch.equal(autocast_output, ordinary_output)
    assert torch.equal(autocast_input.grad, ordinary_gradient)

    valid_tensors = {"weight": original_weight}
    if original_bias is not None:
        valid_tensors["bias"] = original_bias
    invalid_names = {"linear": original_weight}
    if original_bias is not None:
        invalid_names["bias"] = original_bias
    save_file(invalid_names, first_linear.artifact_path)
    with pytest.raises(ValueError, match="layer-bundle linear tensor names differ"):
        first_linear.load_tensors()
    save_file(valid_tensors, first_linear.artifact_path)

    mutated = {"weight": original_weight + 1.0}
    if original_bias is not None:
        mutated["bias"] = original_bias
    save_file(mutated, first_linear.artifact_path)

    assert torch.equal(weight_snapshot, original_weight)
    if bias_snapshot is not None and original_bias is not None:
        assert torch.equal(bias_snapshot, original_bias)
    with pytest.raises(ValueError, match="layer-bundle linear tensor digest mismatch"):
        first_linear.load_tensors()


def test_layer_bundle_t2_evidence_preserves_exact_identity_and_resource_claims() -> None:
    results = CONFIG.parents[1] / "spikes" / "layer-bundle-fp32" / "results"
    exported = json.loads((results / "export-t2.json").read_text(encoding="utf-8"))
    resident = json.loads((results / "resident-t2.json").read_text(encoding="utf-8"))
    direct = json.loads((results / "direct-t2.json").read_text(encoding="utf-8"))
    bundle = json.loads((results / "bundle-t2.json").read_text(encoding="utf-8"))

    expected_base = "47a536cd24b50e7a3bd7a36dc224e2e31774ab0c1c0738df0256e6f579fc15e5"
    expected_gradient = (
        "227b763759a9a63da9eae0ca98af6166a2bccd0dd08212aa668a3b09cdf3b11d"
    )
    assert exported["base_model_sha256"] == expected_base
    assert exported["linear_count"] == 48
    assert exported["artifact_file_count"] == 50
    assert bundle["bundle_manifest_sha256"] == exported["manifest_sha256"]
    assert all(
        result["base_model_sha256"] == expected_base
        and result["loss_sum"] == 4687.0
        and result["gradient_sha256"] == expected_gradient
        for result in (resident, direct, bundle)
    )
    assert bundle["startup_streamed_read_count"] == 0
    assert bundle["startup_streamed_read_bytes"] == 0
    assert bundle["retained_tensor_bytes"] < resident["retained_tensor_bytes"]
    assert bundle["peak_after_build_bytes"] < resident["peak_after_build_bytes"]
    assert bundle["peak_rss_bytes"] < resident["peak_rss_bytes"]
    assert bundle["build_seconds"] < direct["build_seconds"]


def test_int8_t1_homogeneous_trajectory_is_reproducible_and_profile_separate() -> None:
    result_path = (
        CONFIG.parents[1]
        / "spikes"
        / "int8-frozen-linear"
        / "results"
        / "t1-trajectory.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["format"] == "orcacolony_int8_homogeneous_trajectory_v1"
    assert result["campaign_steps"] == 20
    assert result["homogeneous_profile_candidate"] is True
    assert len(result["restart_checks"]) == 10
    assert all(
        check["adapter_metrics"]["max_absolute_error"] == 0.0
        and check["optimizer_metrics"]["max_absolute_error"] == 0.0
        for check in result["restart_checks"]
    )
    assert max(
        step["homogeneous_gradient_metrics"]["relative_l2_error"]
        for step in result["steps"]
    ) < 1e-5
    assert result["final_homogeneous_metrics"]["relative_l2_error"] < 1e-5
    assert result["held_out_improvement"]["fp32"] > 0.0
    assert result["held_out_improvement"]["int8"] > 0.0
    assert result["resident_tensor_bytes"]["int8"] < result[
        "resident_tensor_bytes"
    ]["fp32"]
    assert result["final_adapter_metrics"]["relative_l2_error"] > 1e-2
    assert len(set(result["profiled_final_checkpoint_sha256"].values())) == 3


def test_direct_int8_bundle_startup_evidence_is_exact_and_reduces_t2_peak() -> None:
    results = CONFIG.parents[1] / "spikes" / "int8-frozen-linear" / "results"
    for scale, expected_linears in (("t1", 24), ("t2", 48)):
        converted = json.loads(
            (results / f"startup-converted-{scale}.json").read_text(encoding="utf-8")
        )
        bundle = json.loads(
            (results / f"startup-bundle-{scale}.json").read_text(encoding="utf-8")
        )
        assert converted["format"] == "orcacolony_direct_int8_startup_proof_v1"
        assert bundle["format"] == converted["format"]
        assert bundle["base_model_sha256"] == converted["base_model_sha256"]
        assert bundle["loss_sum"] == converted["loss_sum"]
        assert bundle["gradient_sha256"] == converted["gradient_sha256"]
        assert bundle["retained_tensor_bytes"] == converted["retained_tensor_bytes"]
        assert bundle["bundle_artifact_open_count"] == expected_linears
    converted_t2 = json.loads(
        (results / "startup-converted-t2.json").read_text(encoding="utf-8")
    )
    bundle_t2 = json.loads(
        (results / "startup-bundle-t2.json").read_text(encoding="utf-8")
    )
    assert bundle_t2["peak_after_build_bytes"] < converted_t2[
        "peak_after_build_bytes"
    ]
    assert bundle_t2["peak_rss_bytes"] < converted_t2["peak_rss_bytes"]


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


@pytest.mark.parametrize(
    "artifact,filename",
    [
        ("adapter", "../adapter.safetensors"),
        ("adapter", "nested/adapter.safetensors"),
        ("adapter", r"nested\adapter.safetensors"),
        ("optimizer", "../optimizer.safetensors"),
        ("optimizer", "nested/optimizer.safetensors"),
        ("optimizer", r"C:\outside\optimizer.safetensors"),
    ],
)
def test_lora_checkpoint_rejects_non_basename_artifact_paths(
    tmp_path: Path,
    artifact: str,
    filename: str,
) -> None:
    loaded, checkpoint = _saved_checkpoint(tmp_path)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[artifact]["file"] = filename
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="safe plain basename"):
        load_lora_checkpoint(loaded, checkpoint)


@pytest.mark.parametrize("defect", ["name", "dtype", "shape", "nonfinite"])
def test_lora_checkpoint_rejects_malformed_optimizer_moments(
    tmp_path: Path,
    defect: str,
) -> None:
    loaded, checkpoint = _saved_checkpoint(tmp_path)
    optimizer = dict(load_file(str(checkpoint / "optimizer.safetensors")))
    name = sorted(optimizer)[0]
    tensor = optimizer.pop(name)
    if defect == "name":
        optimizer[f"unexpected.{name}"] = tensor
    elif defect == "dtype":
        optimizer[name] = tensor.to(torch.float64)
    elif defect == "shape":
        optimizer[name] = tensor.reshape(-1)[:-1]
    else:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = torch.nan
        optimizer[name] = tensor
    _rewrite_optimizer(checkpoint, optimizer)

    with pytest.raises(
        ValueError,
        match="tensor set|float32|shape differs|non-finite",
    ):
        load_lora_checkpoint(loaded, checkpoint)


@pytest.mark.parametrize("optimizer_step", [-1, 1.5, True, 2])
def test_lora_checkpoint_rejects_invalid_or_mismatched_optimizer_step(
    tmp_path: Path,
    optimizer_step: object,
) -> None:
    loaded, checkpoint = _saved_checkpoint(tmp_path)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["optimizer_step"] = optimizer_step
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer step"):
        load_lora_checkpoint(loaded, checkpoint)


def _stepped_lora_optimizer():
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    model = build_lora_model(loaded.campaign, loaded.config)
    optimizer = create_adapter_optimizer(model, loaded.campaign.training)
    inputs, targets = fixture_batch(loaded.campaign)
    gradients = compute_adapter_gradients(model, inputs, targets)
    apply_adapter_gradient_step(
        model,
        optimizer,
        gradients.gradients,
        gradients.loss_weight_sum,
        loaded.campaign.training.max_gradient_norm,
    )
    return loaded, model, optimizer


def test_lora_checkpoint_save_rejects_inconsistent_parameter_steps(
    tmp_path: Path,
) -> None:
    loaded, model, optimizer = _stepped_lora_optimizer()
    parameters = list(adapter_named_parameters(model).values())
    optimizer.state[parameters[-1]]["step"] = torch.tensor(2.0)

    with pytest.raises(ValueError, match="optimizer steps do not agree"):
        save_lora_checkpoint(
            loaded,
            model,
            optimizer,
            tmp_path / "invalid",
            step=1,
            dataset_cursor=4,
            loss_history=[1.0],
        )


def test_lora_checkpoint_save_rejects_nonfinite_optimizer_moments(
    tmp_path: Path,
) -> None:
    loaded, model, optimizer = _stepped_lora_optimizer()
    first = next(iter(adapter_named_parameters(model).values()))
    optimizer.state[first]["exp_avg"].reshape(-1)[0] = torch.inf

    with pytest.raises(ValueError, match="non-finite"):
        save_lora_checkpoint(
            loaded,
            model,
            optimizer,
            tmp_path / "invalid",
            step=1,
            dataset_cursor=4,
            loss_history=[1.0],
        )


def test_lora_resume_identity_binds_valid_optimizer_moments(tmp_path: Path) -> None:
    loaded, checkpoint = _saved_checkpoint(tmp_path)
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    original_identity = state["checkpoint_sha256"]
    optimizer = dict(load_file(str(checkpoint / "optimizer.safetensors")))
    name = sorted(optimizer)[0]
    optimizer[name] = optimizer[name].clone()
    optimizer[name].reshape(-1)[0] += 1e-6
    _rewrite_optimizer(checkpoint, optimizer)

    with pytest.raises(ValueError, match="checkpoint identity"):
        load_lora_checkpoint(loaded, checkpoint)

    rewritten = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    assert rewritten["checkpoint_sha256"] == original_identity


@pytest.mark.parametrize(
    ("dataset_cursor", "loss_history", "message"),
    [
        (-1, [1.0], "dataset cursor"),
        (5, [1.0], "dataset cursor"),
        (4, [], "loss history"),
        (4, [float("nan")], "finite"),
        (4, [True], "finite number"),
    ],
)
def test_lora_checkpoint_save_rejects_invalid_trajectory_metadata(
    tmp_path: Path,
    dataset_cursor: int,
    loss_history: list[object],
    message: str,
) -> None:
    loaded, model, optimizer = _stepped_lora_optimizer()

    with pytest.raises(ValueError, match=message):
        save_lora_checkpoint(
            loaded,
            model,
            optimizer,
            tmp_path / "invalid",
            step=1,
            dataset_cursor=dataset_cursor,
            loss_history=loss_history,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_cursor", -1, "dataset cursor"),
        ("dataset_cursor", 5, "dataset cursor"),
        ("loss_history", [], "loss history"),
        ("loss_history", [float("nan")], "finite"),
        ("loss_history", [True], "finite number"),
    ],
)
def test_lora_checkpoint_load_rejects_invalid_trajectory_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    loaded, checkpoint = _saved_checkpoint(tmp_path)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_lora_checkpoint(loaded, checkpoint)


def test_lora_resume_matches_an_uninterrupted_second_step_exactly(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    direct = run_lora_training(loaded, tmp_path / "direct", target_steps=2)
    first = run_lora_training(loaded, tmp_path / "first", target_steps=1)
    resumed = run_lora_training(
        loaded,
        tmp_path / "resumed",
        target_steps=2,
        resume_from=first.checkpoint_dir,
    )

    direct_adapter = load_file(str(direct.checkpoint_dir / "adapter.safetensors"))
    resumed_adapter = load_file(str(resumed.checkpoint_dir / "adapter.safetensors"))
    direct_optimizer = load_file(str(direct.checkpoint_dir / "optimizer.safetensors"))
    resumed_optimizer = load_file(str(resumed.checkpoint_dir / "optimizer.safetensors"))
    assert direct_adapter.keys() == resumed_adapter.keys()
    assert direct_optimizer.keys() == resumed_optimizer.keys()
    assert all(
        torch.equal(direct_adapter[name], resumed_adapter[name])
        for name in direct_adapter
    )
    assert all(
        torch.equal(direct_optimizer[name], resumed_optimizer[name])
        for name in direct_optimizer
    )
    direct_state = json.loads(
        (direct.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    resumed_state = json.loads(
        (resumed.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert direct_state["step"] == resumed_state["step"] == 2
    assert direct_state["optimizer_step"] == resumed_state["optimizer_step"] == 2
    assert direct_state["dataset_cursor"] == resumed_state["dataset_cursor"] == 8
    assert direct_state["loss_history"] == resumed_state["loss_history"]
    assert direct_state["checkpoint_sha256"] == resumed_state["checkpoint_sha256"]
