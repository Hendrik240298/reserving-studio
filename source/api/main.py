from __future__ import annotations

from datetime import datetime, timezone
from typing import NoReturn, Protocol

from fastapi import Depends, FastAPI, HTTPException

from ai.assistant_service import AssistantService
from ai.chat_service import AIChatService
from source.api.adapters.reserving_adapter import (
    InMemoryReservingBackend,
    SessionConflictError,
)
from source.api.schemas import (
    AIChatCreateRequest,
    AIChatCreateResponse,
    AIChatMessageRequest,
    AIChatMessageResponse,
    AIChatSessionResponse,
    DataCompareRequest,
    DataCompareResponse,
    DataViewRequest,
    DataViewResponse,
    DerivedDropScenarioRequest,
    DerivedDropScenarioResponse,
    DiagnosticsIterateRequest,
    DiagnosticsIterateResponse,
    DiagnosticsRequest,
    DiagnosticsResponse,
    HighestA2ADropRequest,
    HighestA2ADropResponse,
    LateEmergenceRequest,
    LateEmergenceResponse,
    LinkRatioRankRequest,
    LinkRatioRankResponse,
    LdfConsistencyRequest,
    LdfConsistencyResponse,
    MovementDiagnosticsRequest,
    MovementDiagnosticsResponse,
    RecalculateRequest,
    RecalculateResponse,
    ReserveChangeRequest,
    ReserveChangeResponse,
    TailEvaluationRequest,
    TailEvaluationResponse,
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

    def get_data_view(self, payload: DataViewRequest) -> DataViewResponse: ...

    def compare_data_views(
        self, payload: DataCompareRequest
    ) -> DataCompareResponse: ...

    def run_movement_diagnostics(
        self,
        payload: MovementDiagnosticsRequest,
    ) -> MovementDiagnosticsResponse: ...

    def run_ldf_consistency(
        self,
        payload: LdfConsistencyRequest,
    ) -> LdfConsistencyResponse: ...

    def project_late_emergence(
        self,
        payload: LateEmergenceRequest,
    ) -> LateEmergenceResponse: ...

    def explain_reserve_change(
        self,
        payload: ReserveChangeRequest,
    ) -> ReserveChangeResponse: ...

    def run_highest_a2a_drop_scenario(
        self,
        payload: HighestA2ADropRequest,
    ) -> HighestA2ADropResponse: ...

    def rank_link_ratios(
        self,
        payload: LinkRatioRankRequest,
    ) -> LinkRatioRankResponse: ...

    def run_derived_drop_scenario(
        self,
        payload: DerivedDropScenarioRequest,
    ) -> DerivedDropScenarioResponse: ...

    def evaluate_tail_fit(
        self,
        payload: TailEvaluationRequest,
    ) -> TailEvaluationResponse: ...


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

    def get_data_view(self, payload: DataViewRequest) -> DataViewResponse:
        raise NotImplementedError("Data view backend not wired yet")

    def compare_data_views(self, payload: DataCompareRequest) -> DataCompareResponse:
        raise NotImplementedError("Data comparison backend not wired yet")

    def run_movement_diagnostics(
        self,
        payload: MovementDiagnosticsRequest,
    ) -> MovementDiagnosticsResponse:
        raise NotImplementedError("Movement diagnostics backend not wired yet")

    def run_ldf_consistency(
        self,
        payload: LdfConsistencyRequest,
    ) -> LdfConsistencyResponse:
        raise NotImplementedError("LDF consistency backend not wired yet")

    def project_late_emergence(
        self,
        payload: LateEmergenceRequest,
    ) -> LateEmergenceResponse:
        raise NotImplementedError("Late emergence backend not wired yet")

    def explain_reserve_change(
        self,
        payload: ReserveChangeRequest,
    ) -> ReserveChangeResponse:
        raise NotImplementedError("Reserve change backend not wired yet")

    def run_highest_a2a_drop_scenario(
        self,
        payload: HighestA2ADropRequest,
    ) -> HighestA2ADropResponse:
        raise NotImplementedError("Highest a2a drop backend not wired yet")

    def rank_link_ratios(
        self,
        payload: LinkRatioRankRequest,
    ) -> LinkRatioRankResponse:
        raise NotImplementedError("Link ratio ranking backend not wired yet")

    def run_derived_drop_scenario(
        self,
        payload: DerivedDropScenarioRequest,
    ) -> DerivedDropScenarioResponse:
        raise NotImplementedError("Derived drop scenario backend not wired yet")

    def evaluate_tail_fit(
        self,
        payload: TailEvaluationRequest,
    ) -> TailEvaluationResponse:
        raise NotImplementedError("Tail evaluation backend not wired yet")


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
    chat_service = AIChatService(
        assistant_factory=lambda: AssistantService.from_backend(backend=backend_impl)
    )

    def get_backend() -> ReservingApiBackend:
        return backend_impl

    def get_chat_service() -> AIChatService:
        return chat_service

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

    @app.post(
        "/v1/data/view",
        response_model=DataViewResponse,
        tags=["Data"],
    )
    def get_data_view(
        payload: DataViewRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> DataViewResponse:
        try:
            return backend_service.get_data_view(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected data view error")

    @app.post(
        "/v1/data/compare",
        response_model=DataCompareResponse,
        tags=["Data"],
    )
    def compare_data_views(
        payload: DataCompareRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> DataCompareResponse:
        try:
            return backend_service.compare_data_views(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected data compare error")

    @app.post(
        "/v1/diagnostics/movement",
        response_model=MovementDiagnosticsResponse,
        tags=["Diagnostics"],
    )
    def run_movement_diagnostics(
        payload: MovementDiagnosticsRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> MovementDiagnosticsResponse:
        try:
            return backend_service.run_movement_diagnostics(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(
            status_code=500, detail="Unexpected movement diagnostics error"
        )

    @app.post(
        "/v1/diagnostics/ldf-consistency",
        response_model=LdfConsistencyResponse,
        tags=["Diagnostics"],
    )
    def run_ldf_consistency(
        payload: LdfConsistencyRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> LdfConsistencyResponse:
        try:
            return backend_service.run_ldf_consistency(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected LDF consistency error")

    @app.post(
        "/v1/diagnostics/late-emergence",
        response_model=LateEmergenceResponse,
        tags=["Diagnostics"],
    )
    def project_late_emergence(
        payload: LateEmergenceRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> LateEmergenceResponse:
        try:
            return backend_service.project_late_emergence(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected late emergence error")

    @app.post(
        "/v1/reserving/explain-change",
        response_model=ReserveChangeResponse,
        tags=["Reserving"],
    )
    def explain_reserve_change(
        payload: ReserveChangeRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> ReserveChangeResponse:
        try:
            return backend_service.explain_reserve_change(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected reserve change error")

    @app.post(
        "/v1/reserving/highest-a2a-drop",
        response_model=HighestA2ADropResponse,
        tags=["Reserving"],
    )
    def run_highest_a2a_drop_scenario(
        payload: HighestA2ADropRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> HighestA2ADropResponse:
        try:
            return backend_service.run_highest_a2a_drop_scenario(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected highest a2a drop error")

    @app.post(
        "/v1/link-ratios/rank",
        response_model=LinkRatioRankResponse,
        tags=["Data"],
    )
    def rank_link_ratios(
        payload: LinkRatioRankRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> LinkRatioRankResponse:
        try:
            return backend_service.rank_link_ratios(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(
            status_code=500, detail="Unexpected link ratio ranking error"
        )

    @app.post(
        "/v1/reserving/derived-drop-scenario",
        response_model=DerivedDropScenarioResponse,
        tags=["Reserving"],
    )
    def run_derived_drop_scenario(
        payload: DerivedDropScenarioRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> DerivedDropScenarioResponse:
        try:
            return backend_service.run_derived_drop_scenario(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(
            status_code=500, detail="Unexpected derived drop scenario error"
        )

    @app.post(
        "/v1/tail/evaluate",
        response_model=TailEvaluationResponse,
        tags=["Diagnostics"],
    )
    def evaluate_tail_fit(
        payload: TailEvaluationRequest,
        backend_service: ReservingApiBackend = Depends(get_backend),
    ) -> TailEvaluationResponse:
        try:
            return backend_service.evaluate_tail_fit(payload)
        except NotImplementedError as error:
            _raise_not_implemented(error)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=500, detail="Unexpected tail evaluation error")

    @app.get("/v1/meta", tags=["System"])
    def meta() -> dict[str, str]:
        return {
            "name": "reserving-api",
            "version": "0.1.0",
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @app.post(
        "/v1/ai/chats",
        response_model=AIChatCreateResponse,
        tags=["AI"],
    )
    def create_chat(
        payload: AIChatCreateRequest,
        chat_service_impl: AIChatService = Depends(get_chat_service),
    ) -> AIChatCreateResponse:
        try:
            session = chat_service_impl.create_chat(
                segment=payload.segment,
                reserving_session_id=payload.reserving_session_id,
            )
            return AIChatCreateResponse(
                chat_id=session.chat_id,
                segment=session.segment,
                reserving_session_id=session.reserving_session_id,
                messages=session.messages,
                tool_events=session.tool_events,
                working_memory=session.working_memory,
                scenario_ledger=session.scenario_ledger,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get(
        "/v1/ai/chats/{chat_id}",
        response_model=AIChatSessionResponse,
        tags=["AI"],
    )
    def get_chat(
        chat_id: str,
        chat_service_impl: AIChatService = Depends(get_chat_service),
    ) -> AIChatSessionResponse:
        session = chat_service_impl.get_chat(chat_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return AIChatSessionResponse(
            chat_id=session.chat_id,
            segment=session.segment,
            reserving_session_id=session.reserving_session_id,
            messages=session.messages,
            tool_events=session.tool_events,
            working_memory=session.working_memory,
            scenario_ledger=session.scenario_ledger,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    @app.post(
        "/v1/ai/chats/{chat_id}/messages",
        response_model=AIChatMessageResponse,
        tags=["AI"],
    )
    def send_chat_message(
        chat_id: str,
        payload: AIChatMessageRequest,
        chat_service_impl: AIChatService = Depends(get_chat_service),
    ) -> AIChatMessageResponse:
        try:
            response = chat_service_impl.send_message(chat_id, payload.content)
            return AIChatMessageResponse(**response)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
