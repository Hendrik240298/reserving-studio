# Architecture Diagrams

This page provides visual maps for the local `source/` reserving app.

## 1) End-to-end workflow

```mermaid
flowchart LR
    A[Config YAML + Session YAML<br/>ConfigManager] --> B[Input Loader]
    B --> C1[ClaimsRepository]
    B --> C2[PremiumInputRepository]
    C2 --> C3[PremiumRepository]
    C1 --> D[ClaimsCollection]
    C3 --> E[Premium DataFrame]
    D --> F[Triangle.from_claims]
    E --> F
    F --> G[Reserving Engine<br/>set_development/set_tail/set_bf/reserve]
    G --> H[ReservingService Payload Builder]
    H --> I[Dashboard UI]
    I --> J[SessionSyncService + tab_sync.js]
    J --> A
    I --> K[FinalizePayload]
    K --> L[Script / ETL / Reporting]
```

## 2) Runtime layering

```mermaid
flowchart TB
    UI[Dash UI<br/>source/dashboard.py]
    Services[source/services/*]
    Domain[source/claims_collection.py<br/>source/triangle.py<br/>source/reserving.py]
    Data[source/input_loader.py<br/>source/claims_repository.py<br/>source/premium_input_repository.py<br/>source/premium_repository.py]
    Config[source/config_manager.py]
    Present[source/presentation/plot_builders.py]

    UI --> Services
    Services --> Domain
    Services --> Present
    Services --> Config
    Domain --> Data
    Data --> Config
```

## 3) Interactive-session contract path

```mermaid
sequenceDiagram
    participant Script
    participant App as app.py
    participant Dash as Dashboard
    participant Ctrl as InteractiveSessionController

    Script->>App: run_interactive_session(reserving, ...)
    App->>Dash: start dashboard thread
    Dash->>Ctrl: publish_latest(params_store, results_store)
    Dash->>Ctrl: finalize(FinalizePayload)
    App->>Ctrl: wait_for_finalize(...)
    Ctrl-->>App: finalized payload
    App-->>Script: FinalizePayload
```

## 4) Cache-key model (high level)

```mermaid
flowchart LR
    P[Params state] --> M[Model Cache Key]
    M --> MP[Model payload cache]
    M --> V[Visual Cache Key]
    V --> VF[Triangle/Emergence figure caches]
    M --> R[Results Cache Key + selected_ultimate_by_uwy]
    R --> RP[Result table payload projection]
```

Notes:

- Cache logic is implemented in `source/services/cache_service.py`.
- Recalculation orchestration and payload assembly are in `source/services/reserving_service.py`.
