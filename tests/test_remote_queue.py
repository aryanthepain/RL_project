"""Unit tests for DualSlotQueueManager and job scheduling."""

import unittest
from src.tqc.remote.queue_manager import DualSlotQueueManager, RemoteJob


class TestRemoteQueue(unittest.TestCase):
    """Test suite for dual-slot queue concurrency, state transitions, and monitoring."""

    def setUp(self):
        self.dispatched = []

    def mock_dispatcher(self, job: RemoteJob) -> bool:
        self.dispatched.append(job.job_id)
        return True

    def test_dual_slot_limit_enforced(self):
        """Test that queue never exceeds max concurrent slots (default 2)."""
        statuses = {
            "job1": "running",
            "job2": "running",
            "job3": "queued",
        }
        manager = DualSlotQueueManager(
            max_concurrent_slots=2,
            status_poller=lambda k_id: statuses.get(k_id, "queued"),
            job_dispatcher=self.mock_dispatcher,
        )

        manager.add_job("job1", "dir1", "HalfCheetah-v4", 42)
        manager.add_job("job2", "dir2", "HalfCheetah-v4", 43)
        manager.add_job("job3", "dir3", "HalfCheetah-v4", 44)

        # Initial step: slots 1 and 2 dispatched
        summary = manager.step_queue()
        self.assertEqual(len(manager.get_active_jobs()), 2)
        self.assertEqual(len(manager.get_pending_jobs()), 1)
        self.assertEqual(self.dispatched, ["job1", "job2"])

    def test_job_completion_releases_slot(self):
        """Test that completing a job releases slot for the next pending job."""
        job_status = {
            "job1": "running",
            "job2": "running",
            "job3": "running",
        }
        manager = DualSlotQueueManager(
            max_concurrent_slots=2,
            status_poller=lambda k_id: job_status.get(k_id, "running"),
            job_dispatcher=self.mock_dispatcher,
        )

        manager.add_job("job1", "dir1", "HalfCheetah-v4", 42)
        manager.add_job("job2", "dir2", "HalfCheetah-v4", 43)
        manager.add_job("job3", "dir3", "HalfCheetah-v4", 44)

        manager.step_queue()
        self.assertEqual(len(manager.get_active_jobs()), 2)

        # job1 finishes
        job_status["job1"] = "complete"
        manager.step_queue()

        # job1 moved to completed, job3 dispatched into vacant slot
        self.assertEqual(len(manager.get_completed_jobs()), 1)
        self.assertEqual(manager.get_completed_jobs()[0].job_id, "job1")
        self.assertEqual(len(manager.get_active_jobs()), 2)
        self.assertEqual(len(manager.get_pending_jobs()), 0)
        self.assertIn("job3", self.dispatched)

    def test_job_failure_handled(self):
        """Test that job failure marks job as failed and frees up slot."""
        job_status = {"job1": "error"}
        manager = DualSlotQueueManager(
            max_concurrent_slots=2,
            status_poller=lambda k_id: job_status.get(k_id, "running"),
            job_dispatcher=self.mock_dispatcher,
        )

        manager.add_job("job1", "dir1", "HalfCheetah-v4", 42)
        manager.step_queue()  # dispatched
        manager.step_queue()  # polls 'error'

        self.assertEqual(len(manager.get_failed_jobs()), 1)
        self.assertEqual(manager.get_failed_jobs()[0].job_id, "job1")
        self.assertEqual(len(manager.get_active_jobs()), 0)

    def test_monitor_until_completion(self):
        """Test monitor loop finishes when all jobs complete."""
        cycle = 0

        def dynamic_poller(k_id):
            nonlocal cycle
            if cycle > 1:
                return "complete"
            return "running"

        manager = DualSlotQueueManager(
            max_concurrent_slots=2,
            status_poller=dynamic_poller,
            job_dispatcher=self.mock_dispatcher,
        )
        manager.add_job("job1", "dir1", "HalfCheetah-v4", 42)

        completed_callbacks = []

        def on_complete(job):
            completed_callbacks.append(job.job_id)

        # Step 1
        manager.step_queue()
        cycle = 2

        res = manager.monitor_until_completion(
            poll_interval=0.01,
            max_iterations=5,
            on_job_complete=on_complete,
        )

        self.assertEqual(res["completed_count"], 1)
        self.assertEqual(completed_callbacks, ["job1"])


if __name__ == "__main__":
    unittest.main()
