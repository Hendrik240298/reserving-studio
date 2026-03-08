from __future__ import annotations

from datetime import datetime, timezone
from typing import NoReturn, Protocol

from fastapi import Depends, FastAPI, HTTPException

from source.api.adapters.reserving_adapter import (
    InMemoryReservingBackend,
    SessionConflictError,
)
from source.api.schemas import (
    DiagnosticsIterateRequest,
    DiagnosticsIterateResponse,
    DiagnosticsRequest,
    DiagnosticsResponse,
    RecalculateRequest,
    RecalculateResponse,
    ResultsResponse,
    SessionSaveRequest,
    SessionSaveResponse,
    SessionStateResponse,
    WorkflowFromDataframesRequest,
    WorkflowInitializationResponse,
)


class ReservingApiBackend(Protocol):
    def create_workflow_from_dataframes(
        self,
        payload: WorkflowFromDataframesRequest,
    ) -> WorkflowInitializationResponse: ...

    def get_session(self, segment: str) -> SessionStateResponse | None: ...

    def save_session(
        self,
        segment: str,
        payload: SessionSaveRequest,
    ) -> SessionSaveResponse: ...

    def recalculate(self, payload: RecalculateRequest) -> RecalculateResponse: ...

    def run_diagnostics(self, payload: DiagnosticsRequest) -> DiagnosticsResponse: ...

    def iterate_diagnostics(
        self,
        payload: DiagnosticsIterateRequest,
    ) -> DiagnosticsIterateResponse: ...

    def get_results(self, session_id: str) -> ResultsResponse | None: ...


class NotImplementedBackend:
    """Scaffold backend adapter.

    Replace this with an implementation that delegates to existing
    `source/services/*` modules.
    """

    def create_workflow_from_dataframes(
        self,
        payload: WorkflowFromDataframesRequest,
    ) -> WorkflowInitializationResponse:
        raise NotImplementedError("Workflow initialization backend not wired yet")

    def get_session(self, segment: str) -> SessionStateResponse | None:
        raise NotImplementedError("Session read backend not wired yet")

    def save_session(
        self,
        segment: str,
        payload: SessionSaveRequest,
    ) -> SessionSaveResponse:
        raise NotImplementedError("Session save backend not wired yet")

    def recalculate(self, payload: RecalculateRequest) -> RecalculateResponse:
        raise NotImplementedError("Reserving recalc backend not wired yet")

    def run_diagnostics(self, payload: DiagnosticsRequest) -> DiagnosticsResponse:
        raise NotImplementedError("Diagnostics backend not wired yet")

    def get_results(self, session_id: str) -> ResultsResponse | None:
        raise NotImplementedError("Results backend not wired yet")

    def iterate_diagnostics(
        self,
        payload: DiagnosticsIterateRequest,
    ) -> DiagnosticsIterateResponse:
        raise NotImplementedError("Diagnostics iteration backend not wired yet")


def _raise_not_implemented(error: NotImplementedError) -> NoReturn:
    raise HTTPException(status_code=501, detail=str(error)) from error


def create_app(backend: ReservingApiBackend | None = None) -> FastAPI:
    app = FastAPI(
        title="Reserving API",
        version="0.1.0",
        description=(
            "Contract-first API for reserving workflows, deterministic diagnostics, "
            "and AI/GUI integration."
        ),
    )
    backend_impl = backend or InMemoryReservingBackend()

    def get_backend() -> ReservingApiBackend:
        return backend_impl

    @app.get("/healthz", tags=["System"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/workflows/from-dataframes",
        response_model=WorkflowInitializationResponse,
        tags=["Workflows"],
    )
    def create_workflow_from_dataframes(
        payload: WorkflowFromDataframesRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> WorkflowInitializationResponse:
        try:
            return backend_service.create_workflow_from_dataframes(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected workflow error")

    @app.get(
        "/v1/sessions/{segment}",
        response_model=SessionStateResponse,
        tags=["Sessions"],
    )
    def get_session(
        segment: str,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> SessionStateResponse:
        response: SessionStateResponse | None = None
        try:
            response = backend_service.get_session(segment)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if response is None:
            raise HTTPException(status_code=404, detail="Segment session not found")
        return response

    @app.post(
        "/v1/sessions/{segment}/save",
        response_model=SessionSaveResponse,
        tags=["Sessions"],
    )
    def save_session(
        segment: str,
        payload: SessionSaveRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> SessionSaveResponse:
        try:
            return backend_service.save_session(segment, payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except SessionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected session save error")

    @app.post(
        "/v1/reserving/recalculate",
        response_model=RecalculateResponse,
        tags=["Reserving"],
    )
    def recalculate(
        payload: RecalculateRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> RecalculateResponse:
        try:
            return backend_service.recalculate(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected recalculate error")

    @app.post(
        "/v1/diagnostics/run",
        response_model=DiagnosticsResponse,
        tags=["Diagnostics"],
    )
    def run_diagnostics(
        payload: DiagnosticsRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> DiagnosticsResponse:
        try:
            return backend_service.run_diagnostics(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected diagnostics error")

    @app.post(
        "/v1/diagnostics/iterate",
        response_model=DiagnosticsIterateResponse,
        tags=["Diagnostics"],
    )
    def iterate_diagnostics(
        payload: DiagnosticsIterateRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> DiagnosticsIterateResponse:
        try:
            return backend_service.iterate_diagnostics(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(
            status_code=500, detail="Unexpected diagnostics iteration error"
        )

    @app.get(
        "/v1/results/{session_id}",
        response_model=ResultsResponse,
        tags=["Results"],
    )
    def get_results(
        session_id: str,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> ResultsResponse:
        response: ResultsResponse | None = None
        try:
            response = backend_service.get_results(session_id)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if response is None:
            raise HTTPException(status_code=404, detail="Session results not found")
        return response

    @app.get("/v1/meta", tags=["System"])
    def meta() -> dict[str, str]:
        return {
            "name": "reserving-api",
            "version": "0.1.0",
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    return app


app = create_app()
