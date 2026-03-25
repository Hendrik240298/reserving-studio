from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable


class BackgroundRecalcService:
    def __init__(self, worker: Callable[[dict], dict]) -> None:
        self._worker = worker
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recalc")
        self._lock = Lock()
        self._current_future: Future | None = None
        self._current_job: dict | None = None
        self._pending_job: dict | None = None
        self._latest_request_id = 0
        self._last_error: str | None = None

    def _start_job_unlocked(self, job: dict) -> None:
        self._current_job = job
        self._current_future = self._executor.submit(self._worker, job)
        self._last_error = None
        logging.info(
            "Background recalc started request=%s",
            job.get("request_id"),
        )

    def submit(self, job: dict) -> dict:
        request_id = int(job.get("request_id", 0))
        with self._lock:
            self._latest_request_id = max(self._latest_request_id, request_id)
            if self._current_future is not None and not self._current_future.done():
                self._pending_job = job
                logging.info(
                    "Background recalc queued request=%s after request=%s",
                    request_id,
                    self._current_job.get("request_id") if self._current_job else None,
                )
                return {
                    "state": "queued",
                    "request_id": request_id,
                    "running_request_id": (
                        self._current_job.get("request_id")
                        if self._current_job
                        else None
                    ),
                }

            self._start_job_unlocked(job)
            return {"state": "running", "request_id": request_id}

    def poll(self) -> dict:
        with self._lock:
            current_future = self._current_future
            current_job = self._current_job
            pending_job = self._pending_job

            if current_future is None:
                if pending_job is not None:
                    self._pending_job = None
                    self._start_job_unlocked(pending_job)
                    return {
                        "state": "running",
                        "request_id": pending_job.get("request_id"),
                    }
                if self._last_error is not None:
                    return {"state": "error", "error": self._last_error}
                return {"state": "idle"}

            if not current_future.done():
                state = "queued" if pending_job is not None else "running"
                return {
                    "state": state,
                    "request_id": current_job.get("request_id")
                    if current_job
                    else None,
                    "pending_request_id": (
                        pending_job.get("request_id")
                        if pending_job is not None
                        else None
                    ),
                }

            self._current_future = None
            self._current_job = None
            pending_to_start = self._pending_job
            self._pending_job = None

        try:
            result = current_future.result()
        except Exception as exc:
            logging.exception("Background recalc failed")
            with self._lock:
                if pending_to_start is not None:
                    self._start_job_unlocked(pending_to_start)
                    return {
                        "state": "running",
                        "request_id": pending_to_start.get("request_id"),
                    }
                self._last_error = str(exc)
                return {"state": "error", "error": str(exc)}

        completed_request_id = int(result.get("request_id", 0))

        with self._lock:
            if pending_to_start is not None:
                self._start_job_unlocked(pending_to_start)
                if completed_request_id < self._latest_request_id:
                    logging.info(
                        "Discarded stale background recalc result request=%s latest=%s",
                        completed_request_id,
                        self._latest_request_id,
                    )
                    return {
                        "state": "running",
                        "request_id": pending_to_start.get("request_id"),
                    }

            if completed_request_id < self._latest_request_id:
                logging.info(
                    "Ignored stale background recalc result request=%s latest=%s",
                    completed_request_id,
                    self._latest_request_id,
                )
                return {"state": "idle"}

            self._last_error = None
            return {
                "state": "completed",
                "request_id": completed_request_id,
                "result": result,
            }
