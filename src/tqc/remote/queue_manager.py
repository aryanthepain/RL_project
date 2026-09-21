"""Dual-slot queue manager for Kaggle GPU workloads enforcing concurrency constraints."""

import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class QueueError(Exception):
    """Raised when queue management or job dispatching encounters an error."""
    pass


@dataclass
class RemoteJob:
    """Represents an experiment scheduled for remote execution."""
    job_id: str
    kernel_dir: str
    env_id: str
    seed: int
    kernel_id: Optional[str] = None
    accelerator: Optional[str] = "NvidiaTeslaT4"
    status: str = "pending"  # pending, queued, running, complete, error
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualSlotQueueManager:
    """Manages execution queue enforcing Kaggle's maximum concurrent GPU limit (default 2)."""

    def __init__(
        self,
        max_concurrent_slots: int = 2,
        status_poller: Optional[Callable[[str], str]] = None,
        job_dispatcher: Optional[Callable[[RemoteJob], bool]] = None,
    ):
        """Initialize queue manager.

        Args:
            max_concurrent_slots: Maximum concurrent running kernels (Kaggle GPU limit is 2).
            status_poller: Optional custom callable to query kernel status (for testing/mocking).
            job_dispatcher: Optional custom callable to dispatch/push kernel (for testing/mocking).
        """
        self.max_concurrent_slots = max_concurrent_slots
        self._poller = status_poller or self._default_poll_status
        self._dispatcher = job_dispatcher or self._default_dispatch_job

        self.pending_jobs: List[RemoteJob] = []
        self.active_jobs: List[RemoteJob] = []
        self.completed_jobs: List[RemoteJob] = []
        self.failed_jobs: List[RemoteJob] = []

    def add_job(
        self,
        job_id: str,
        kernel_dir: str,
        env_id: str,
        seed: int,
        kernel_id: Optional[str] = None,
        accelerator: Optional[str] = "NvidiaTeslaT4",
    ) -> RemoteJob:
        """Enqueue a new experiment job."""
        job = RemoteJob(
            job_id=job_id,
            kernel_dir=kernel_dir,
            env_id=env_id,
            seed=seed,
            kernel_id=kernel_id or job_id,
            accelerator=accelerator,
        )
        self.pending_jobs.append(job)
        return job

    def get_active_jobs(self) -> List[RemoteJob]:
        """Return currently active (queued or running) jobs."""
        return list(self.active_jobs)

    def get_pending_jobs(self) -> List[RemoteJob]:
        """Return jobs waiting to be dispatched."""
        return list(self.pending_jobs)

    def get_completed_jobs(self) -> List[RemoteJob]:
        """Return successfully completed jobs."""
        return list(self.completed_jobs)

    def get_failed_jobs(self) -> List[RemoteJob]:
        """Return failed jobs."""
        return list(self.failed_jobs)

    def _default_poll_status(self, kernel_id: str) -> str:
        """Poll kernel status via Kaggle CLI."""
        cmd = ["kaggle", "kernels", "status", kernel_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.lower()
            if "complete" in output:
                return "complete"
            elif "running" in output:
                return "running"
            elif "queued" in output:
                return "queued"
            elif "error" in output or "failed" in output:
                return "error"
            return "queued"
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout).lower()
            if "not found" in err:
                return "error"
            return "queued"
        except Exception:
            return "error"

    def _default_dispatch_job(self, job: RemoteJob) -> bool:
        """Push kernel directory via Kaggle CLI."""
        cmd = ["kaggle", "kernels", "push", "-p", job.kernel_dir]
        if job.accelerator:
            cmd.extend(["--accelerator", job.accelerator])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except Exception as e:
            job.error_message = str(e)
            return False

    def poll_job_status(self, kernel_id: str) -> str:
        """Query status of a specific kernel."""
        return self._poller(kernel_id)

    def step_queue(self) -> Dict[str, Any]:
        """Advance the queue: check active jobs, clean completed ones, and dispatch pending."""
        still_active: List[RemoteJob] = []

        # 1. Update status of active jobs
        for job in self.active_jobs:
            current_status = self._poller(job.kernel_id)
            job.status = current_status

            if current_status == "complete":
                job.finished_at = datetime.now().isoformat()
                self.completed_jobs.append(job)
                print(f"[QueueManager] Job {job.job_id} ({job.kernel_id}) COMPLETED successfully.")
            elif current_status == "error":
                job.finished_at = datetime.now().isoformat()
                self.failed_jobs.append(job)
                print(f"[QueueManager] Job {job.job_id} ({job.kernel_id}) FAILED with status 'error'.")
            else:
                # Still queued or running
                still_active.append(job)

        self.active_jobs = still_active

        # 2. Fill available slots from pending queue
        slots_available = self.max_concurrent_slots - len(self.active_jobs)
        while slots_available > 0 and self.pending_jobs:
            next_job = self.pending_jobs.pop(0)
            next_job.started_at = datetime.now().isoformat()
            print(f"[QueueManager] Dispatching job {next_job.job_id} (Slots in use: {len(self.active_jobs) + 1}/{self.max_concurrent_slots})...")

            success = self._dispatcher(next_job)
            if success:
                next_job.status = "queued"
                self.active_jobs.append(next_job)
                slots_available -= 1
            else:
                next_job.status = "error"
                next_job.finished_at = datetime.now().isoformat()
                self.failed_jobs.append(next_job)
                print(f"[QueueManager] Dispatch failed for job {next_job.job_id}: {next_job.error_message}")

        return {
            "active": [j.to_dict() for j in self.active_jobs],
            "pending": [j.to_dict() for j in self.pending_jobs],
            "completed": [j.to_dict() for j in self.completed_jobs],
            "failed": [j.to_dict() for j in self.failed_jobs],
        }

    def monitor_until_completion(
        self,
        poll_interval: int = 30,
        max_iterations: Optional[int] = None,
        on_job_complete: Optional[Callable[[RemoteJob], None]] = None,
    ) -> Dict[str, Any]:
        """Poll and manage queue until all pending and active jobs are finished.

        Args:
            poll_interval: Seconds between polling cycles.
            max_iterations: Optional maximum loop cycles (useful for testing).
            on_job_complete: Callback function invoked when a job finishes.

        Returns:
            Dictionary with final counts of completed and failed jobs.
        """
        iteration = 0
        previously_completed = set(j.job_id for j in self.completed_jobs)

        while self.pending_jobs or self.active_jobs:
            iteration += 1
            print(f"\n[QueueManager Poll #{iteration}] Active: {len(self.active_jobs)}, Pending: {len(self.pending_jobs)}, Completed: {len(self.completed_jobs)}, Failed: {len(self.failed_jobs)}")
            summary = self.step_queue()

            # Trigger callbacks for newly completed jobs
            if on_job_complete:
                for job in self.completed_jobs:
                    if job.job_id not in previously_completed:
                        previously_completed.add(job.job_id)
                        on_job_complete(job)

            if not self.pending_jobs and not self.active_jobs:
                break

            if max_iterations and iteration >= max_iterations:
                print("[QueueManager] Reached max iterations limit.")
                break

            time.sleep(poll_interval)

        return {
            "completed_count": len(self.completed_jobs),
            "failed_count": len(self.failed_jobs),
            "remaining_active": len(self.active_jobs),
            "remaining_pending": len(self.pending_jobs),
        }
