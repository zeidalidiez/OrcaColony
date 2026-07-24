from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor
from torch.nn import functional as F

from .coordinator import _tensor_metrics
from .participants import ParticipantRegistry, load_participants
from .reference import (
    CampaignConfig,
    _create_optimizer,
    _load_checkpoint,
    _save_checkpoint,
    _sha256_file,
    build_model,
    fixture_batch,
    run_training,
)


@dataclass(frozen=True)
class LeasedGradient:
    assignment_id: str
    lease_token: str
    checkpoint_sha256: str
    loss_sum: float
    loss_weight_sum: int
    safetensors: bytes
    runtime_backend: str


RUNTIME_BACKENDS = frozenset(
    {"burn-ndarray-f32", "burn-webgpu-f32", "python-oracle-f32"}
)


@dataclass(frozen=True)
class WorkReceipt:
    assignment_id: str
    accepted: bool
    step_complete: bool
    step: int
    model_sha256: str | None
    gradient_metrics: Mapping[str, float | int | str]
    checkpoint_metrics: Mapping[str, float | int | str]


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _revision(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _campaign_payload(campaign: CampaignConfig) -> dict[str, object]:
    return {
        "campaign": dict(campaign.campaign),
        "model": asdict(campaign.model),
        "training": asdict(campaign.training),
    }


class GlobalStepCoordinator:
    def __init__(
        self,
        campaign: CampaignConfig,
        state_dir: Path,
        state: dict[str, object],
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.initial_model_path = state_dir / "model.safetensors"
        self.base_checkpoint_dir = state_dir / "base-checkpoint"
        self.oracle_dir = state_dir / "oracle-gradients"
        self.results_dir = state_dir / "results"
        self.reference_dir = state_dir / "reference-step-1"
        self.checkpoint_dir = state_dir / "checkpoint"
        self._state = state
        self.participants = ParticipantRegistry.from_payload(
            state["participants"],  # type: ignore[arg-type]
            campaign_id=str(campaign.campaign["id"]),
        )
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        worker_count: int,
        participants: ParticipantRegistry,
        lease_seconds: int = 60,
        resume_from: str | Path | None = None,
    ) -> GlobalStepCoordinator:
        if worker_count < 2:
            raise ValueError("multi-worker proof requires at least two workers")
        if campaign.training.batch_size % worker_count:
            raise ValueError("training batch size must be divisible by worker count")
        if lease_seconds <= 0:
            raise ValueError("lease duration must be positive")
        if participants.campaign_id != campaign.campaign["id"]:
            raise ValueError("participant registry campaign mismatch")

        state_dir = Path(state_dir)
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"coordinator state directory is not empty: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "oracle-gradients").mkdir()
        (state_dir / "results").mkdir()

        base_step = 0
        dataset_cursor = 0
        loss_history: list[float] = []
        if resume_from is None:
            initial_model = build_model(campaign)
            save_safetensors_file(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in sorted(initial_model.state_dict().items())
                },
                str(state_dir / "model.safetensors"),
            )
        else:
            source_checkpoint = Path(resume_from)
            (
                initial_model,
                _,
                base_step,
                dataset_cursor,
                loss_history,
            ) = _load_checkpoint(campaign, source_checkpoint)
            base_checkpoint_dir = state_dir / "base-checkpoint"
            base_checkpoint_dir.mkdir()
            source_state = json.loads(
                (source_checkpoint / "state.json").read_text(encoding="utf-8")
            )
            for filename in (
                source_state["model"]["file"],
                source_state["optimizer"]["file"],
                "state.json",
            ):
                shutil.copy2(source_checkpoint / filename, base_checkpoint_dir / filename)
            shutil.copy2(
                base_checkpoint_dir / source_state["model"]["file"],
                state_dir / "model.safetensors",
            )
        checkpoint_sha256 = _sha256_file(state_dir / "model.safetensors")
        run_training(
            campaign,
            state_dir / "reference-step-1",
            target_steps=base_step + 1,
            resume_from=(state_dir / "base-checkpoint" if resume_from is not None else None),
        )

        inputs, targets = fixture_batch(campaign, dataset_cursor)
        rows_per_assignment = campaign.training.batch_size // worker_count
        assignments: list[dict[str, object]] = []
        for index in range(worker_count):
            start = index * rows_per_assignment
            end = start + rows_per_assignment
            assignment_inputs = inputs[start:end].contiguous()
            assignment_targets = targets[start:end].contiguous()
            model = build_model(campaign)
            model.load_state_dict(initial_model.state_dict())
            loss_sum = F.cross_entropy(
                model(assignment_inputs).reshape(-1, campaign.model.vocabulary_size),
                assignment_targets.reshape(-1),
                reduction="sum",
            )
            loss_sum.backward()
            gradients = {
                name: parameter.grad.detach().cpu().contiguous()
                for name, parameter in sorted(model.named_parameters())
                if parameter.grad is not None
            }
            basis = {
                "campaign_id": campaign.campaign["id"],
                "checkpoint_sha256": checkpoint_sha256,
                "model": {
                    "vocab_size": campaign.model.vocabulary_size,
                    "context_length": campaign.model.context_length,
                    "d_model": campaign.model.width,
                    "num_heads": campaign.model.heads,
                    "num_layers": campaign.model.layers,
                    "d_ff": campaign.model.mlp_width,
                },
                "global_step": base_step,
                "data_range": [dataset_cursor + start, dataset_cursor + end],
                "input_ids": assignment_inputs.reshape(-1).tolist(),
                "input_shape": list(assignment_inputs.shape),
                "target_ids": assignment_targets.reshape(-1).tolist(),
                "target_shape": list(assignment_targets.shape),
                "loss_weight_sum": assignment_targets.numel(),
            }
            assignment_id = hashlib.sha256(
                json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            oracle_file = f"{assignment_id}.safetensors"
            save_safetensors_file(
                gradients,
                str(state_dir / "oracle-gradients" / oracle_file),
            )
            assignments.append(
                {
                    "assignment_id": assignment_id,
                    **basis,
                    "parameter_count": campaign.model.parameters,
                    "expected_loss_sum": float(loss_sum.detach()),
                    "oracle_file": oracle_file,
                    "state": "open",
                    "attempt": 0,
                    "leased_by": None,
                    "contributor_id": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "result_file": None,
                    "accepted_loss_sum": None,
                    "runtime_backend": None,
                    "gradient_metrics": None,
                }
            )

        state: dict[str, object] = {
            "format": "orcacolony_global_step_v1",
            "campaign_id": campaign.campaign["id"],
            "campaign_revision": _revision(_campaign_payload(campaign)),
            "participants": participants.as_payload(),
            "participants_revision": participants.revision,
            "checkpoint_sha256": checkpoint_sha256,
            "worker_count": worker_count,
            "lease_seconds": lease_seconds,
            "state": "waiting_for_results",
            "step": base_step,
            "base_step": base_step,
            "dataset_cursor": dataset_cursor,
            "loss_history": loss_history,
            "has_base_checkpoint": resume_from is not None,
            "result_protocol_revision": 2,
            "assignments": assignments,
            "model_sha256": None,
            "checkpoint_metrics": None,
        }
        _atomic_json(state_dir / "global-state.json", state)
        coordinator = cls(campaign, state_dir, state)
        coordinator._write_campaign_lock()
        coordinator._write_accepted_ledger()
        return coordinator

    @classmethod
    def load(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        participants: ParticipantRegistry,
    ) -> GlobalStepCoordinator:
        state_dir = Path(state_dir)
        state = json.loads((state_dir / "global-state.json").read_text(encoding="utf-8"))
        if state.get("format") != "orcacolony_global_step_v1":
            raise ValueError("unsupported global-step state format")
        if state.get("campaign_id") != campaign.campaign["id"]:
            raise ValueError("global-step campaign does not match configuration")
        campaign_revision = _revision(_campaign_payload(campaign))
        migrated = "participants_revision" not in state
        if migrated:
            state.setdefault("base_step", 0)
            state.setdefault("dataset_cursor", 0)
            state.setdefault("loss_history", [])
            state.setdefault("has_base_checkpoint", False)
            state["campaign_revision"] = campaign_revision
            state["participants"] = participants.as_payload()
            state["participants_revision"] = participants.revision
            for assignment in state["assignments"]:
                worker_id = assignment.get("leased_by")
                if worker_id is None:
                    assignment["contributor_id"] = None
                    continue
                participant = participants.participant_for_worker(str(worker_id))
                if participant is None:
                    raise ValueError(
                        f"existing worker is not allowlisted: {worker_id}"
                    )
                assignment["contributor_id"] = participant.contributor_id
            _atomic_json(state_dir / "global-state.json", state)
        elif state.get("participants_revision") != participants.revision:
            raise ValueError("participant revision mismatch")
        if state.get("campaign_revision") != campaign_revision:
            raise ValueError("campaign revision mismatch")
        protocol_migrated = (
            "result_protocol_revision" not in state
            and state["state"] != "step_complete"
        )
        if protocol_migrated:
            state["result_protocol_revision"] = 2
        if _sha256_file(state_dir / "model.safetensors") != state["checkpoint_sha256"]:
            raise ValueError("global-step checkpoint digest mismatch")
        coordinator = cls(campaign, state_dir, state)
        state_changed = protocol_migrated
        for assignment in coordinator.assignments:
            if "model" not in assignment:
                assignment["model"] = {
                    "vocab_size": campaign.model.vocabulary_size,
                    "context_length": campaign.model.context_length,
                    "d_model": campaign.model.width,
                    "num_heads": campaign.model.heads,
                    "num_layers": campaign.model.layers,
                    "d_ff": campaign.model.mlp_width,
                }
                state_changed = True
            if "runtime_backend" not in assignment:
                assignment["runtime_backend"] = (
                    "legacy-unknown" if assignment["state"] == "accepted" else None
                )
                state_changed = True
        if state_changed:
            coordinator._write_state()
        lock_path = state_dir / "campaign-lock.json"
        if migrated or protocol_migrated:
            coordinator._write_campaign_lock()
        elif not lock_path.exists() or json.loads(lock_path.read_text(encoding="utf-8")) != coordinator._campaign_lock_payload():
            raise ValueError("campaign lock mismatch")
        if state["state"] != "step_complete" and coordinator._all_accepted():
            with coordinator._lock:
                coordinator._state["state"] = "ready_to_finalize"
                coordinator._write_state()
                coordinator._finalize_locked()
        elif state["state"] == "step_complete":
            checkpoint_state = json.loads(
                (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
            )
            if checkpoint_state["model"]["sha256"] != state["model_sha256"]:
                raise ValueError("completed global-step checkpoint identity mismatch")
            model_path = coordinator.checkpoint_dir / checkpoint_state["model"]["file"]
            optimizer_path = (
                coordinator.checkpoint_dir / checkpoint_state["optimizer"]["file"]
            )
            if (
                _sha256_file(model_path) != checkpoint_state["model"]["sha256"]
                or _sha256_file(optimizer_path)
                != checkpoint_state["optimizer"]["sha256"]
            ):
                raise ValueError("completed global-step checkpoint digest mismatch")
        coordinator._write_accepted_ledger()
        return coordinator

    @property
    def assignments(self) -> list[dict[str, object]]:
        return self._state["assignments"]  # type: ignore[return-value]

    def _write_state(self) -> None:
        _atomic_json(self.state_dir / "global-state.json", self._state)

    def _campaign_lock_payload(self) -> dict[str, object]:
        return {
            "format": "orcacolony_campaign_lock_v1",
            "campaign_id": self._state["campaign_id"],
            "campaign_revision": self._state["campaign_revision"],
            "participants_revision": self._state["participants_revision"],
            "checkpoint_sha256": self._state["checkpoint_sha256"],
            "global_step": self._state["base_step"],
            "assignment_protocol_revision": 1,
            "result_protocol_revision": self._state.get(
                "result_protocol_revision", 1
            ),
        }

    def _write_campaign_lock(self) -> None:
        _atomic_json(
            self.state_dir / "campaign-lock.json",
            self._campaign_lock_payload(),
        )

    def _write_accepted_ledger(self) -> None:
        entries: list[dict[str, object]] = []
        for assignment in sorted(
            self.assignments,
            key=lambda value: value["data_range"][0],
        ):
            if assignment["state"] != "accepted":
                continue
            contributor_id = str(assignment["contributor_id"])
            participant = next(
                value
                for value in self.participants.participants
                if value.contributor_id == contributor_id
            )
            public_credit = (
                {"display_name": participant.display_name}
                if participant.public_credit
                else None
            )
            entries.append(
                {
                    "assignment_id": assignment["assignment_id"],
                    "global_step": assignment["global_step"],
                    "data_range": assignment["data_range"],
                    "attempt": assignment["attempt"],
                    "worker_id": assignment["leased_by"],
                    "contributor_id": contributor_id,
                    "public_credit": public_credit,
                    "loss_sum": assignment["accepted_loss_sum"],
                    "loss_weight_sum": assignment["loss_weight_sum"],
                    "runtime_backend": assignment["runtime_backend"],
                }
            )
        _atomic_json(
            self.state_dir / "accepted-work.json",
            {
                "format": "orcacolony_accepted_work_v1",
                "campaign_id": self._state["campaign_id"],
                "participants_revision": self._state["participants_revision"],
                "entries": entries,
            },
        )

    def _all_accepted(self) -> bool:
        return all(assignment["state"] == "accepted" for assignment in self.assignments)

    def oracle_gradient_path(self, assignment_id: str) -> Path:
        assignment = self._assignment(assignment_id)
        return self.oracle_dir / str(assignment["oracle_file"])

    def _assignment(self, assignment_id: str) -> dict[str, object]:
        for assignment in self.assignments:
            if assignment["assignment_id"] == assignment_id:
                return assignment
        raise ValueError("unknown assignment")

    def _public_assignment(self, assignment: Mapping[str, object]) -> dict[str, object]:
        assignment_id = str(assignment["assignment_id"])
        return {
            "format": "orcacolony_assignment_v1",
            "campaign_id": assignment["campaign_id"],
            "assignment_id": assignment_id,
            "checkpoint_sha256": assignment["checkpoint_sha256"],
            "model": assignment["model"],
            "global_step": assignment["global_step"],
            "data_range": assignment["data_range"],
            "input_ids": assignment["input_ids"],
            "input_shape": assignment["input_shape"],
            "target_ids": assignment["target_ids"],
            "target_shape": assignment["target_shape"],
            "loss_weight_sum": assignment["loss_weight_sum"],
            "parameter_count": assignment["parameter_count"],
            "expected_loss_sum": assignment["expected_loss_sum"],
            "attempt": assignment["attempt"],
            "lease_token": assignment["lease_token"],
            "lease_expires_at": assignment["lease_expires_at"],
            "model_url": "/api/v1/artifacts/model.safetensors",
            "oracle_gradient_url": f"/api/v1/oracle/{assignment_id}.safetensors",
            "result_url": f"/api/v1/results/{assignment_id}",
            "result_protocol_revision": self._state.get(
                "result_protocol_revision", 1
            ),
            "runtime_backends": sorted(RUNTIME_BACKENDS),
        }

    def lease(
        self,
        worker_id: str,
        worker_token: str | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        if not worker_id:
            raise ValueError("worker identity is required")
        participant = self.participants.participant_for_worker(worker_id)
        if participant is None:
            raise ValueError("worker is not allowlisted for this campaign")
        if not self.participants.credential_is_valid(
            participant,
            worker_id,
            worker_token,
        ):
            raise ValueError("worker credential is invalid")
        now = time.time() if now is None else now
        with self._lock:
            if self._state["state"] == "step_complete":
                raise ValueError("global step is already complete")
            for assignment in self.assignments:
                if (
                    assignment["state"] == "leased"
                    and assignment["leased_by"] == worker_id
                    and float(assignment["lease_expires_at"]) > now
                ):
                    return self._public_assignment(assignment)

            available = [
                assignment
                for assignment in self.assignments
                if assignment["state"] == "open"
                or (
                    assignment["state"] == "leased"
                    and float(assignment["lease_expires_at"]) <= now
                )
            ]
            if not available:
                raise ValueError("no assignment is currently available")
            assignment = min(available, key=lambda value: value["data_range"][0])
            attempt = int(assignment["attempt"]) + 1
            lease_token = hashlib.sha256(
                f"{assignment['assignment_id']}:{worker_id}:{attempt}".encode("utf-8")
            ).hexdigest()
            assignment.update(
                {
                    "state": "leased",
                    "attempt": attempt,
                    "leased_by": worker_id,
                    "contributor_id": participant.contributor_id,
                    "lease_token": lease_token,
                    "lease_expires_at": now + int(self._state["lease_seconds"]),
                }
            )
            self._write_state()
            return self._public_assignment(assignment)

    def accept(
        self,
        submission: LeasedGradient,
        now: float | None = None,
        finalize: bool = True,
    ) -> WorkReceipt:
        now = time.time() if now is None else now
        with self._lock:
            assignment = self._assignment(submission.assignment_id)
            if assignment["state"] == "accepted":
                raise ValueError("assignment result was already accepted")
            if (
                assignment["state"] != "leased"
                or assignment["lease_token"] != submission.lease_token
                or float(assignment["lease_expires_at"]) <= now
            ):
                raise ValueError("stale lease attempt")
            if submission.checkpoint_sha256 != self._state["checkpoint_sha256"]:
                raise ValueError("checkpoint identity does not match")
            if submission.loss_weight_sum != assignment["loss_weight_sum"]:
                raise ValueError("loss weight does not match assignment")
            if not math.isfinite(submission.loss_sum):
                raise ValueError("loss sum must be finite")
            if submission.runtime_backend not in RUNTIME_BACKENDS:
                raise ValueError("runtime backend is not supported")
            expected_loss = float(assignment["expected_loss_sum"])
            if abs(submission.loss_sum - expected_loss) / abs(expected_loss) > 0.002:
                raise ValueError("loss sum is outside the M2 tolerance")

            gradients = load_safetensors(submission.safetensors)
            expected_gradients = load_safetensors_file(
                str(self.oracle_gradient_path(submission.assignment_id))
            )
            gradient_metrics = _tensor_metrics(expected_gradients, gradients)
            if float(gradient_metrics["cosine_similarity"]) < 0.999:
                raise ValueError("gradient cosine similarity is outside the M2 tolerance")
            if float(gradient_metrics["relative_l2_error"]) > 0.01:
                raise ValueError("gradient relative L2 error is outside the M2 tolerance")

            result_file = f"{submission.assignment_id}.safetensors"
            _atomic_bytes(self.results_dir / result_file, submission.safetensors)
            assignment.update(
                {
                    "state": "accepted",
                    "result_file": result_file,
                    "accepted_loss_sum": submission.loss_sum,
                    "runtime_backend": submission.runtime_backend,
                    "gradient_metrics": gradient_metrics,
                    "lease_token": None,
                    "lease_expires_at": None,
                }
            )
            self._state["state"] = (
                "ready_to_finalize" if self._all_accepted() else "waiting_for_results"
            )
            self._write_state()
            self._write_accepted_ledger()
            if self._all_accepted() and finalize:
                self._finalize_locked()
            return self._receipt(assignment)

    def _finalize_locked(self) -> None:
        if self._state.get("has_base_checkpoint", False):
            (
                model,
                optimizer,
                base_step,
                dataset_cursor,
                loss_history,
            ) = _load_checkpoint(self.campaign, self.base_checkpoint_dir)
        else:
            model = build_model(self.campaign)
            model.load_state_dict(load_safetensors_file(str(self.initial_model_path)))
            optimizer = _create_optimizer(model, self.campaign.training)
            base_step = 0
            dataset_cursor = 0
            loss_history = []
        aggregate = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in model.named_parameters()
        }
        total_loss_sum = 0.0
        total_loss_weight = 0
        for assignment in sorted(self.assignments, key=lambda value: value["data_range"][0]):
            gradients = load_safetensors_file(
                str(self.results_dir / str(assignment["result_file"]))
            )
            for name in aggregate:
                aggregate[name].add_(gradients[name])
            total_loss_sum += float(assignment["accepted_loss_sum"])
            total_loss_weight += int(assignment["loss_weight_sum"])

        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.grad = aggregate[name].div(total_loss_weight)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.campaign.training.max_gradient_norm,
        )
        optimizer.step()
        next_step = base_step + 1
        next_cursor = (
            dataset_cursor + self.campaign.training.batch_size
        ) % self.campaign.training.dataset_sequences
        checkpoint = _save_checkpoint(
            self.campaign,
            model,
            optimizer,
            self.checkpoint_dir,
            step=next_step,
            dataset_cursor=next_cursor,
            loss_history=[*loss_history, total_loss_sum / total_loss_weight],
        )
        checkpoint_metrics = _tensor_metrics(
            load_safetensors_file(str(self.reference_dir / "model.safetensors")),
            load_safetensors_file(str(self.checkpoint_dir / "model.safetensors")),
        )
        self._state.update(
            {
                "state": "step_complete",
                "step": next_step,
                "model_sha256": checkpoint.model_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "loss_sum": total_loss_sum,
                "loss_weight_sum": total_loss_weight,
            }
        )
        self._write_state()
        _atomic_json(
            self.state_dir / "global-receipt.json",
            {
                "format": "orcacolony_global_step_receipt_v1",
                "state": "step_complete",
                "step": next_step,
                "model_sha256": checkpoint.model_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "assignments": [
                    {
                        "assignment_id": assignment["assignment_id"],
                        "attempt": assignment["attempt"],
                        "data_range": assignment["data_range"],
                        "gradient_metrics": assignment["gradient_metrics"],
                        "runtime_backend": assignment["runtime_backend"],
                    }
                    for assignment in self.assignments
                ],
            },
        )

    def _receipt(self, assignment: Mapping[str, object]) -> WorkReceipt:
        step_complete = self._state["state"] == "step_complete"
        return WorkReceipt(
            assignment_id=str(assignment["assignment_id"]),
            accepted=True,
            step_complete=step_complete,
            step=int(self._state["step"]),
            model_sha256=(
                str(self._state["model_sha256"]) if step_complete else None
            ),
            gradient_metrics=assignment["gradient_metrics"],  # type: ignore[arg-type]
            checkpoint_metrics=(
                self._state["checkpoint_metrics"] if step_complete else {}
            ),  # type: ignore[arg-type]
        )

    def status(self) -> dict[str, object]:
        return {
            "state": self._state["state"],
            "campaign_id": self._state["campaign_id"],
            "checkpoint_sha256": self._state["checkpoint_sha256"],
            "step": self._state["step"],
            "model_sha256": self._state["model_sha256"],
            "checkpoint_metrics": self._state["checkpoint_metrics"],
            "loss_sum": self._state.get("loss_sum"),
            "loss_weight_sum": self._state.get("loss_weight_sum"),
            "assignments": [
                {
                    "assignment_id": assignment["assignment_id"],
                    "data_range": assignment["data_range"],
                    "state": assignment["state"],
                    "attempt": assignment["attempt"],
                    "leased_by": assignment["leased_by"],
                }
                for assignment in self.assignments
            ],
        }


class _GlobalStepHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: object,
        coordinator: GlobalStepCoordinator,
        directory: str,
        **kwargs: object,
    ) -> None:
        self.coordinator = coordinator
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(
        self,
        payload: Mapping[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_artifact(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "artifact is unavailable"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/v1/assignment":
            worker_id = parse_qs(parsed.query).get("worker_id", [""])[0]
            try:
                assignment = self.coordinator.lease(
                    worker_id,
                    worker_token=self.headers.get("X-Orca-Worker-Token"),
                )
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
                return
            self._send_json(assignment)
            return
        if path == "/api/v1/status":
            self._send_json(self.coordinator.status())
            return
        if path == "/api/v1/artifacts/model.safetensors":
            self._send_artifact(self.coordinator.initial_model_path)
            return
        if path.startswith("/api/v1/oracle/") and path.endswith(".safetensors"):
            assignment_id = path.removeprefix("/api/v1/oracle/").removesuffix(
                ".safetensors"
            )
            try:
                artifact = self.coordinator.oracle_gradient_path(assignment_id)
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact(artifact)
            return
        if path == "/api/v1/checkpoint/model.safetensors":
            self._send_artifact(self.coordinator.checkpoint_dir / "model.safetensors")
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        prefix = "/api/v1/results/"
        if not path.startswith(prefix):
            self._send_json({"error": "unknown result endpoint"}, HTTPStatus.NOT_FOUND)
            return
        assignment_id = path.removeprefix(prefix)
        try:
            maximum_length = (
                self.coordinator.oracle_gradient_path(assignment_id).stat().st_size + 1024
            )
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > maximum_length:
                raise ValueError("gradient payload length is invalid")
            submission = LeasedGradient(
                assignment_id=assignment_id,
                lease_token=self.headers["X-Orca-Lease-Token"],
                checkpoint_sha256=self.headers["X-Orca-Checkpoint-Sha256"],
                loss_sum=float(self.headers["X-Orca-Loss-Sum"]),
                loss_weight_sum=int(self.headers["X-Orca-Loss-Weight-Sum"]),
                safetensors=self.rfile.read(content_length),
                runtime_backend=self.headers["X-Orca-Runtime-Backend"],
            )
            receipt = self.coordinator.accept(submission)
        except (KeyError, TypeError, ValueError, SafetensorError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "accepted": receipt.accepted,
                "assignment_id": receipt.assignment_id,
                "step_complete": receipt.step_complete,
                "step": receipt.step,
                "model_sha256": receipt.model_sha256,
                "gradient_metrics": receipt.gradient_metrics,
                "checkpoint_metrics": receipt.checkpoint_metrics,
                "checkpoint_url": (
                    "/api/v1/checkpoint/model.safetensors"
                    if receipt.step_complete
                    else None
                ),
            }
        )


def create_http_server(
    coordinator: GlobalStepCoordinator,
    browser_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    browser_root = Path(browser_root).resolve()
    if not (browser_root / "index.html").is_file():
        raise ValueError(f"browser root does not contain index.html: {browser_root}")

    def handler(*args: object, **kwargs: object) -> _GlobalStepHandler:
        return _GlobalStepHandler(
            *args,
            coordinator=coordinator,
            directory=str(browser_root),
            **kwargs,
        )

    return ThreadingHTTPServer((host, port), handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OrcaColony M2 multi-worker global-step proof"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    from .reference import load_campaign

    args = _build_parser().parse_args()
    campaign = load_campaign(args.config)
    participants = load_participants(
        args.participants,
        campaign_id=str(campaign.campaign["id"]),
    )
    state_path = args.state / "global-state.json"
    coordinator = (
        GlobalStepCoordinator.load(campaign, args.state, participants=participants)
        if state_path.is_file()
        else GlobalStepCoordinator.create(
            campaign,
            args.state,
            worker_count=args.workers,
            participants=participants,
            lease_seconds=args.lease_seconds,
            resume_from=args.resume_from,
        )
    )
    server = create_http_server(
        coordinator,
        args.browser_root,
        host=args.host,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign["id"],
                "state": coordinator.status()["state"],
                "url_template": (
                    f"http://{args.host}:{server.server_port}/"
                    "?worker=<worker-id>#token=<worker-token>"
                ),
                "workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
