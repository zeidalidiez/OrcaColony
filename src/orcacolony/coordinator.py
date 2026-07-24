from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from .reference import (
    CampaignConfig,
    TrainingResult,
    _create_optimizer,
    _save_checkpoint,
    build_model,
    export_fixture,
    run_training,
)


@dataclass(frozen=True)
class SubmittedGradient:
    assignment_id: str
    checkpoint_sha256: str
    loss_sum: float
    loss_weight_sum: int
    safetensors: bytes


@dataclass(frozen=True)
class AcceptedGradient:
    step: int
    checkpoint_dir: Path
    model_sha256: str
    gradient_metrics: Mapping[str, float | int | str]
    checkpoint_metrics: Mapping[str, float | int | str]


def _tensor_metrics(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> dict[str, float | int | str]:
    expected_names = sorted(expected)
    actual_names = sorted(actual)
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_names))
        raise ValueError(f"tensor names differ: missing={missing}, unexpected={unexpected}")

    squared_error = 0.0
    squared_expected = 0.0
    squared_actual = 0.0
    dot = 0.0
    max_absolute_error = 0.0
    value_count = 0
    worst_tensor = ""

    for name in expected_names:
        left = expected[name].detach().cpu()
        right = actual[name].detach().cpu()
        if right.dtype != torch.float32:
            raise ValueError(f"{name}: expected float32, got {right.dtype}")
        if right.shape != left.shape:
            raise ValueError(
                f"{name}: shape differs: expected={tuple(left.shape)}, actual={tuple(right.shape)}"
            )
        if not bool(torch.isfinite(right).all()):
            raise ValueError(f"{name}: contains non-finite values")

        left64 = left.to(torch.float64)
        right64 = right.to(torch.float64)
        difference = right64 - left64
        tensor_max = float(difference.abs().max()) if difference.numel() else 0.0
        if tensor_max > max_absolute_error:
            max_absolute_error = tensor_max
            worst_tensor = name
        squared_error += float(torch.sum(difference * difference))
        squared_expected += float(torch.sum(left64 * left64))
        squared_actual += float(torch.sum(right64 * right64))
        dot += float(torch.sum(left64 * right64))
        value_count += difference.numel()

    relative_l2_error = math.sqrt(squared_error / squared_expected) if squared_expected else 0.0
    denominator = math.sqrt(squared_expected * squared_actual)
    cosine_similarity = dot / denominator if denominator else 1.0
    return {
        "tensor_count": len(expected_names),
        "value_count": value_count,
        "cosine_similarity": cosine_similarity,
        "relative_l2_error": relative_l2_error,
        "max_absolute_error": max_absolute_error,
        "worst_tensor": worst_tensor,
    }


class ConnectedCoordinator:
    def __init__(
        self,
        campaign: CampaignConfig,
        state_dir: Path,
        assignment: dict[str, object],
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.fixture_dir = state_dir / "fixture"
        self.reference_dir = state_dir / "reference-step-1"
        self.checkpoint_dir = state_dir / "checkpoint"
        self.assignment = assignment
        self._model = model
        self._optimizer = optimizer
        self._accepted: AcceptedGradient | None = None
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
    ) -> ConnectedCoordinator:
        state_dir = Path(state_dir)
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"coordinator state directory is not empty: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
        fixture_dir = state_dir / "fixture"
        export_fixture(campaign, fixture_dir)
        run_training(campaign, state_dir / "reference-step-1", target_steps=1)

        fixture = json.loads((fixture_dir / "fixture.json").read_text(encoding="utf-8"))
        assignment_basis = {
            "campaign_id": fixture["campaign_id"],
            "checkpoint_sha256": fixture["files"]["model.safetensors"],
            "model": fixture["model"],
            "step": 0,
            "input_ids": fixture["input_ids"],
            "input_shape": fixture["input_shape"],
            "target_ids": fixture["target_ids"],
            "target_shape": fixture["target_shape"],
            "loss_weight_sum": fixture["loss_weight_sum"],
        }
        assignment_id = hashlib.sha256(
            json.dumps(
                assignment_basis,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assignment = {
            "format": "orcacolony_assignment_v1",
            "assignment_id": assignment_id,
            **assignment_basis,
            "parameter_count": fixture["parameter_count"],
            "expected_loss_sum": fixture["loss_sum"],
            "model_url": "/api/v1/artifacts/model.safetensors",
            "oracle_gradient_url": "/api/v1/artifacts/gradients.safetensors",
            "result_url": f"/api/v1/results/{assignment_id}",
        }
        model = build_model(campaign)
        optimizer = _create_optimizer(model, campaign.training)
        return cls(campaign, state_dir, assignment, model, optimizer)

    def status(self) -> dict[str, object]:
        if self._accepted is None:
            return {
                "state": "waiting_for_result",
                "campaign_id": self.assignment["campaign_id"],
                "assignment_id": self.assignment["assignment_id"],
                "step": 0,
            }
        return {
            "state": "step_complete",
            "campaign_id": self.assignment["campaign_id"],
            "assignment_id": self.assignment["assignment_id"],
            "step": self._accepted.step,
            "model_sha256": self._accepted.model_sha256,
            "gradient_metrics": self._accepted.gradient_metrics,
            "checkpoint_metrics": self._accepted.checkpoint_metrics,
        }

    def accept(self, submission: SubmittedGradient) -> AcceptedGradient:
        with self._lock:
            if self._accepted is not None:
                raise ValueError("assignment result has already been accepted")
            if submission.assignment_id != self.assignment["assignment_id"]:
                raise ValueError("assignment identity does not match")
            if submission.checkpoint_sha256 != self.assignment["checkpoint_sha256"]:
                raise ValueError("checkpoint identity does not match")
            if submission.loss_weight_sum != self.assignment["loss_weight_sum"]:
                raise ValueError("loss weight does not match assignment")
            if not math.isfinite(submission.loss_sum):
                raise ValueError("loss sum must be finite")

            expected_loss = float(self.assignment["expected_loss_sum"])
            loss_relative_error = abs(submission.loss_sum - expected_loss) / abs(expected_loss)
            if loss_relative_error > 0.002:
                raise ValueError("loss sum is outside the M1 tolerance")

            gradients = load_safetensors(submission.safetensors)
            expected_gradients = load_safetensors_file(
                str(self.fixture_dir / "gradients.safetensors")
            )
            gradient_metrics = _tensor_metrics(expected_gradients, gradients)
            if float(gradient_metrics["cosine_similarity"]) < 0.999:
                raise ValueError("gradient cosine similarity is outside the M1 tolerance")
            if float(gradient_metrics["relative_l2_error"]) > 0.01:
                raise ValueError("gradient relative L2 error is outside the M1 tolerance")

            self._optimizer.zero_grad(set_to_none=True)
            for name, parameter in self._model.named_parameters():
                parameter.grad = gradients[name].to(parameter.dtype).clone()
                parameter.grad.div_(submission.loss_weight_sum)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.campaign.training.max_gradient_norm,
            )
            self._optimizer.step()

            checkpoint: TrainingResult = _save_checkpoint(
                self.campaign,
                self._model,
                self._optimizer,
                self.checkpoint_dir,
                step=1,
                dataset_cursor=self.campaign.training.batch_size,
                loss_history=[submission.loss_sum / submission.loss_weight_sum],
            )
            expected_model = load_safetensors_file(
                str(self.reference_dir / "model.safetensors")
            )
            actual_model = load_safetensors_file(
                str(self.checkpoint_dir / "model.safetensors")
            )
            checkpoint_metrics = _tensor_metrics(expected_model, actual_model)
            accepted = AcceptedGradient(
                step=1,
                checkpoint_dir=self.checkpoint_dir,
                model_sha256=checkpoint.model_sha256,
                gradient_metrics=gradient_metrics,
                checkpoint_metrics=checkpoint_metrics,
            )
            receipt = {
                "format": "orcacolony_result_receipt_v1",
                "accepted": True,
                "assignment_id": submission.assignment_id,
                "step": accepted.step,
                "model_sha256": accepted.model_sha256,
                "gradient_metrics": accepted.gradient_metrics,
                "checkpoint_metrics": accepted.checkpoint_metrics,
            }
            (self.state_dir / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._accepted = accepted
            return accepted


class _CoordinatorHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: object,
        coordinator: ConnectedCoordinator,
        directory: str,
        **kwargs: object,
    ) -> None:
        self.coordinator = coordinator
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(self, payload: Mapping[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
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
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/v1/assignment":
            self._send_json(self.coordinator.assignment)
            return
        if path == "/api/v1/status":
            self._send_json(self.coordinator.status())
            return
        if path == "/api/v1/artifacts/model.safetensors":
            self._send_artifact(self.coordinator.fixture_dir / "model.safetensors")
            return
        if path == "/api/v1/artifacts/gradients.safetensors":
            self._send_artifact(self.coordinator.fixture_dir / "gradients.safetensors")
            return
        if path == "/api/v1/checkpoint/model.safetensors":
            self._send_artifact(self.coordinator.checkpoint_dir / "model.safetensors")
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        expected_path = f"/api/v1/results/{self.coordinator.assignment['assignment_id']}"
        if path != expected_path:
            self._send_json({"error": "unknown result endpoint"}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            maximum_length = (
                self.coordinator.fixture_dir / "gradients.safetensors"
            ).stat().st_size + 1024
            if content_length <= 0 or content_length > maximum_length:
                raise ValueError("gradient payload length is invalid")
            submission = SubmittedGradient(
                assignment_id=str(self.coordinator.assignment["assignment_id"]),
                checkpoint_sha256=self.headers["X-Orca-Checkpoint-Sha256"],
                loss_sum=float(self.headers["X-Orca-Loss-Sum"]),
                loss_weight_sum=int(self.headers["X-Orca-Loss-Weight-Sum"]),
                safetensors=self.rfile.read(content_length),
            )
            accepted = self.coordinator.accept(submission)
        except (KeyError, TypeError, ValueError, SafetensorError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "accepted": True,
                "assignment_id": submission.assignment_id,
                "step": accepted.step,
                "model_sha256": accepted.model_sha256,
                "gradient_metrics": accepted.gradient_metrics,
                "checkpoint_metrics": accepted.checkpoint_metrics,
                "checkpoint_url": "/api/v1/checkpoint/model.safetensors",
            }
        )


def create_http_server(
    coordinator: ConnectedCoordinator,
    browser_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    browser_root = Path(browser_root).resolve()
    if not (browser_root / "index.html").is_file():
        raise ValueError(f"browser root does not contain index.html: {browser_root}")

    def handler(*args: object, **kwargs: object) -> _CoordinatorHandler:
        return _CoordinatorHandler(
            *args,
            coordinator=coordinator,
            directory=str(browser_root),
            **kwargs,
        )

    return ThreadingHTTPServer((host, port), handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded OrcaColony M1b connected-worker coordinator"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    from .reference import load_campaign

    args = _build_parser().parse_args()
    coordinator = ConnectedCoordinator.create(load_campaign(args.config), args.state)
    server = create_http_server(
        coordinator,
        args.browser_root,
        host=args.host,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "campaign_id": coordinator.assignment["campaign_id"],
                "assignment_id": coordinator.assignment["assignment_id"],
                "url": f"http://{args.host}:{server.server_port}/?connected=1",
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
