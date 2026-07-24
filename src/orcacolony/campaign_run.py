from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from pathlib import Path

from .multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    WorkReceipt,
    _atomic_json,
    _campaign_payload,
    _revision,
    create_http_server,
)
from .participants import ParticipantRegistry, load_participants
from .reference import CampaignConfig, load_campaign


class CampaignCoordinator:
    def __init__(
        self,
        campaign: CampaignConfig,
        state_dir: Path,
        participants: ParticipantRegistry,
        state: dict[str, object],
        current: GlobalStepCoordinator,
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.participants = participants
        self._state = state
        self._current = current
        self._lock = threading.RLock()
        self.rounds_dir = state_dir / "rounds"
        self.checkpoints_dir = state_dir / "checkpoints"

    @classmethod
    def create(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        participants: ParticipantRegistry,
        worker_count: int,
        target_steps: int,
        lease_seconds: int = 120,
    ) -> CampaignCoordinator:
        if target_steps < 1:
            raise ValueError("campaign target steps must be positive")
        if participants.campaign_id != campaign.campaign["id"]:
            raise ValueError("participant campaign does not match configuration")
        state_dir = Path(state_dir)
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"campaign state directory is not empty: {state_dir}")
        rounds_dir = state_dir / "rounds"
        checkpoints_dir = state_dir / "checkpoints"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        current_relative = Path("rounds") / "round-00000000"
        current = GlobalStepCoordinator.create(
            campaign,
            state_dir / current_relative,
            worker_count=worker_count,
            participants=participants,
            lease_seconds=lease_seconds,
        )
        state: dict[str, object] = {
            "format": "orcacolony_campaign_state_v1",
            "campaign_id": campaign.campaign["id"],
            "campaign_revision": _revision(_campaign_payload(campaign)),
            "participants_revision": participants.revision,
            "worker_count": worker_count,
            "lease_seconds": lease_seconds,
            "target_steps": target_steps,
            "completed_steps": 0,
            "state": "campaign_running",
            "current_round": current_relative.as_posix(),
            "checkpoints": [],
            "last_checkpoint_metrics": None,
        }
        coordinator = cls(campaign, state_dir, participants, state, current)
        coordinator._write_state()
        coordinator._write_lock()
        coordinator._write_ledger()
        return coordinator

    @classmethod
    def load(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        participants: ParticipantRegistry,
    ) -> CampaignCoordinator:
        state_dir = Path(state_dir)
        state = json.loads(
            (state_dir / "campaign-state.json").read_text(encoding="utf-8")
        )
        if state.get("format") != "orcacolony_campaign_state_v1":
            raise ValueError("unsupported campaign state format")
        if state.get("campaign_id") != campaign.campaign["id"]:
            raise ValueError("campaign state does not match configuration")
        if state.get("campaign_revision") != _revision(_campaign_payload(campaign)):
            raise ValueError("campaign revision mismatch")
        if state.get("participants_revision") != participants.revision:
            raise ValueError("participant revision mismatch")
        current_path = state_dir / str(state["current_round"])
        current = GlobalStepCoordinator.load(
            campaign,
            current_path,
            participants=participants,
        )
        coordinator = cls(campaign, state_dir, participants, state, current)
        lock_path = state_dir / "campaign-lock.json"
        if (
            not lock_path.is_file()
            or json.loads(lock_path.read_text(encoding="utf-8"))
            != coordinator._lock_payload()
        ):
            raise ValueError("campaign lock mismatch")
        coordinator._advance_if_ready()
        coordinator._write_ledger()
        return coordinator

    @property
    def initial_model_path(self) -> Path:
        return self._current.initial_model_path

    @property
    def checkpoint_dir(self) -> Path:
        checkpoints = self._state["checkpoints"]
        if checkpoints:
            return self.state_dir / str(checkpoints[-1]["path"])
        return self._current.checkpoint_dir

    def oracle_gradient_path(self, assignment_id: str) -> Path:
        return self._current.oracle_gradient_path(assignment_id)

    def lease(
        self,
        worker_id: str,
        worker_token: str | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._state["state"] == "campaign_complete":
                raise ValueError("campaign target is complete")
            return self._current.lease(
                worker_id,
                worker_token=worker_token,
                now=now,
            )

    def accept(
        self,
        submission: LeasedGradient,
        now: float | None = None,
    ) -> WorkReceipt:
        with self._lock:
            if self._state["state"] == "campaign_complete":
                raise ValueError("campaign target is complete")
            receipt = self._current.accept(submission, now=now)
            if receipt.step_complete:
                self._advance_if_ready()
            return receipt

    def _write_state(self) -> None:
        _atomic_json(self.state_dir / "campaign-state.json", self._state)

    def _lock_payload(self) -> dict[str, object]:
        return {
            "format": "orcacolony_campaign_run_lock_v1",
            "campaign_id": self._state["campaign_id"],
            "campaign_revision": self._state["campaign_revision"],
            "participants_revision": self._state["participants_revision"],
            "worker_count": self._state["worker_count"],
            "target_steps": self._state["target_steps"],
            "assignment_protocol_revision": 1,
            "result_protocol_revision": 2,
        }

    def _write_lock(self) -> None:
        _atomic_json(self.state_dir / "campaign-lock.json", self._lock_payload())

    def _version_checkpoint(self, step: int) -> Path:
        destination = self.checkpoints_dir / f"step-{step:08d}"
        source = self._current.checkpoint_dir
        source_state = json.loads((source / "state.json").read_text(encoding="utf-8"))
        if destination.exists():
            destination_state = json.loads(
                (destination / "state.json").read_text(encoding="utf-8")
            )
            if destination_state != source_state:
                raise ValueError("versioned checkpoint does not match completed round")
            return destination

        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
        return destination

    def _advance_if_ready(self) -> None:
        if self._current.status()["state"] != "step_complete":
            return
        completed_step = int(self._current.status()["step"])
        checkpoint = self._version_checkpoint(completed_step)
        checkpoints: list[dict[str, object]] = self._state["checkpoints"]  # type: ignore[assignment]
        if not any(int(value["step"]) == completed_step for value in checkpoints):
            checkpoints.append(
                {
                    "step": completed_step,
                    "path": checkpoint.relative_to(self.state_dir).as_posix(),
                    "round": self._state["current_round"],
                }
            )
        self._state["completed_steps"] = completed_step
        self._state["last_checkpoint_metrics"] = self._current.status()[
            "checkpoint_metrics"
        ]

        if completed_step >= int(self._state["target_steps"]):
            self._state["state"] = "campaign_complete"
            self._write_state()
            self._write_ledger()
            return

        next_relative = Path("rounds") / f"round-{completed_step:08d}"
        next_path = self.state_dir / next_relative
        if (next_path / "global-state.json").is_file():
            next_round = GlobalStepCoordinator.load(
                self.campaign,
                next_path,
                participants=self.participants,
            )
        else:
            next_round = GlobalStepCoordinator.create(
                self.campaign,
                next_path,
                worker_count=int(self._state["worker_count"]),
                participants=self.participants,
                lease_seconds=int(self._state["lease_seconds"]),
                resume_from=checkpoint,
            )
        self._current = next_round
        self._state["current_round"] = next_relative.as_posix()
        self._state["state"] = "campaign_running"
        self._write_state()
        self._write_ledger()

    def _write_ledger(self) -> None:
        entries: list[dict[str, object]] = []
        for checkpoint in self._state["checkpoints"]:
            round_ledger = self.state_dir / str(checkpoint["round"]) / "accepted-work.json"
            payload = json.loads(round_ledger.read_text(encoding="utf-8"))
            for entry in payload["entries"]:
                entries.append({**entry, "checkpoint_step": checkpoint["step"]})
        entries.sort(key=lambda value: (value["checkpoint_step"], value["data_range"][0]))
        _atomic_json(
            self.state_dir / "accepted-work.json",
            {
                "format": "orcacolony_campaign_accepted_work_v1",
                "campaign_id": self._state["campaign_id"],
                "participants_revision": self._state["participants_revision"],
                "entries": entries,
            },
        )

    def status(self) -> dict[str, object]:
        current = self._current.status()
        return {
            "state": self._state["state"],
            "campaign_id": self._state["campaign_id"],
            "completed_steps": self._state["completed_steps"],
            "target_steps": self._state["target_steps"],
            "current_step": current["step"],
            "checkpoint_metrics": self._state["last_checkpoint_metrics"],
            "current_round": current,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent multi-step OrcaColony campaign"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    campaign = load_campaign(args.config)
    participants = load_participants(
        args.participants,
        campaign_id=str(campaign.campaign["id"]),
    )
    if (args.state / "campaign-state.json").is_file():
        coordinator = CampaignCoordinator.load(
            campaign,
            args.state,
            participants=participants,
        )
        if coordinator.status()["target_steps"] != args.target_steps:
            raise ValueError("target steps do not match the campaign lock")
    else:
        coordinator = CampaignCoordinator.create(
            campaign,
            args.state,
            participants=participants,
            worker_count=args.workers,
            target_steps=args.target_steps,
            lease_seconds=args.lease_seconds,
        )
    server = create_http_server(
        coordinator,  # type: ignore[arg-type]
        args.browser_root,
        host=args.host,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign["id"],
                "status": coordinator.status(),
                "url_template": (
                    f"http://{args.host}:{server.server_port}/"
                    "?worker=<worker-id>#token=<worker-token>"
                ),
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
