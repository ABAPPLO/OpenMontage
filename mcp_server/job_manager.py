"""In-process async job manager for long-running tool execution.

OpenMontage tools are synchronous and blocking (FFmpeg/Remotion can run for
minutes). For long jobs, an MCP client doesn't want to hold a single call open
the whole time — it wants to submit, get a job id, and poll. This module
provides that: ``submit`` schedules the work on the event loop's thread pool and
returns immediately with a job id; ``get_status`` reports progress; the result is
held in memory once the job finishes.

Scope / limits (documented honestly):
  - Jobs live in the server process's memory. A server restart loses in-flight
    and completed jobs. This is the right first cut for a single-process stdio
    MCP server; durable/persistent jobs would need a backing store.
  - There's no cross-process work queue — one server runs one job's tool at a
    time within a worker thread, but multiple jobs can be in-flight concurrently
    (bounded by the default executor's thread count).
  - Jobs are NOT replayed from checkpoints — that's the pipeline's job, not the
    job manager's. The job manager only wraps a single tool.execute() call.

The manager is a singleton (``jobs``) so the FastMCP handlers share one registry.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from tools.base_tool import BaseTool, ToolResult

from mcp_server.execution import serialize_result


class JobStatus(str, Enum):
    PENDING = "pending"        # queued, not yet started
    RUNNING = "running"        # tool.execute() in progress
    SUCCEEDED = "succeeded"    # finished, result available
    FAILED = "failed"          # finished, error captured
    CANCELLED = "cancelled"    # removed before completion (not yet implemented)


@dataclass
class Job:
    """One unit of asynchronous tool work."""

    job_id: str
    tool_name: str
    inputs: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None  # serialized ToolResult on success
    error: Optional[str] = None              # message on failure (not result.error)
    progress: float = 0.0                    # 0.0–1.0; coarse (0 → 1) until tools stream

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot for MCP transport. Omits None result until done."""
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at,
        }
        if self.started_at is not None:
            out["started_at"] = self.started_at
        if self.finished_at is not None:
            out["finished_at"] = self.finished_at
            # Elapsed wall time is convenient for clients deciding whether to retry.
            out["elapsed_seconds"] = round(self.finished_at - self.started_at, 2)
        if self.result is not None:
            out["result"] = self.result
        if self.error is not None:
            out["error"] = self.error
        return out


class JobManager:
    """Tracks async tool jobs; runs each on the event loop's thread pool."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(
        self,
        tool: BaseTool,
        inputs: dict[str, Any],
        *,
        started_hook: Optional[Callable[[Job], None]] = None,
    ) -> Job:
        """Schedule a tool for async execution; return its Job immediately.

        ``started_hook`` is invoked when the work actually begins (optional; used
        by handlers to emit an MCP progress notification).
        """
        job = Job(job_id=_new_job_id(), tool_name=tool.name, inputs=dict(inputs))
        async with self._lock:
            self._jobs[job.job_id] = job

        task = asyncio.create_task(self._run(job, tool, inputs, started_hook))
        self._tasks[job.job_id] = task
        return job

    async def _run(
        self,
        job: Job,
        tool: BaseTool,
        inputs: dict[str, Any],
        started_hook: Optional[Callable[[Job], None]],
    ) -> None:
        """Execute the tool off the event loop and record the outcome."""
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        if started_hook is not None:
            try:
                started_hook(job)
            except Exception:
                # A notification hook must never fail the job itself.
                pass

        import anyio  # lazy import, same bridge as execute_tool_async

        def _do_work() -> ToolResult:
            try:
                return tool.execute(inputs)
            except Exception as exc:  # noqa: BLE001 — any raise -> failed result
                return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

        try:
            result = await anyio.to_thread.run_sync(_do_work)
        except Exception as exc:  # noqa: BLE001 — thread-pool-level failure
            job.status = JobStatus.FAILED
            job.error = f"job execution error: {type(exc).__name__}: {exc}"
        else:
            job.progress = 1.0
            job.finished_at = time.time()
            if result.success:
                job.status = JobStatus.SUCCEEDED
                job.result = serialize_result(result)
            else:
                # A tool that reports failure is still a *completed* job whose
                # result carries the error — distinguish from a job that crashed.
                job.status = JobStatus.SUCCEEDED
                job.result = serialize_result(result)
        finally:
            if job.finished_at is None:
                job.finished_at = time.time()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """All jobs, newest-first by creation time."""
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def await_completion(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        """Block until a job reaches a terminal state (for tests + clients that
        want to await rather than poll). Returns the Job or None if not found.

        ``timeout`` in seconds; raises asyncio.TimeoutError if exceeded.
        """
        task = self._tasks.get(job_id)
        job = self._jobs.get(job_id)
        if task is None or job is None:
            return None
        if timeout is not None:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task
        return self._jobs.get(job_id)

    def clear(self) -> None:
        """Drop all job records (tests / cleanup). Does not cancel running tasks."""
        self._jobs.clear()
        self._tasks.clear()


def _new_job_id() -> str:
    """Short, unique, human-readable job id."""
    return "job_" + uuid.uuid4().hex[:12]


# Singleton shared by the FastMCP handlers.
jobs = JobManager()
