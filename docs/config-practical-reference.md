# Config Practical Reference

This page explains the YAML keys you are most likely to adjust as an actuarial user.

## Core keys

```yaml
paths:
  results: results/
  plots: plots/
  data: data/
  sessions: sessions/
first date: 1900
last date: "December 2006"
segment: quarterly
granularity: quarterly
session:
  path: sessions/quarterly.yml
```

- `segment`: run identifier used for session scoping.
- `granularity`: `quarterly` or `yearly` aggregation of inputs.
- `session.path`: where current parameter state is persisted.
- `first date` / `last date`: bounds used by claim-side validation logic.

## Workflow dataset presets

Used when no custom `workflow.input` is provided.

```yaml
workflow:
  dataset: quarterly   # quarterly | clrd
  quarterly_premium_csv: data/quarterly_premium.csv
  clrd_lob: comauto
```

## Custom input mode (`workflow.input`)

Define source by dataset (`claims`, `premium`) and type (`csv`, `sql`).

### CSV example

```yaml
workflow:
  input:
    claims:
      source: csv
      path: data/claims.csv
      column_map:
        claim_id: id
        underwriting_year: uw_year
        valuation_date: period
        paid_movement: paid
        reserve_movement: outstanding
    premium:
      source: csv
      path: data/premium.csv
      column_map:
        underwriting_year: uw_year
        valuation_date: period
        earned_premium: Premium_selected
```

### SQL example

```yaml
workflow:
  input:
    sql:
      driver: ODBC Driver 18 for SQL Server
      server: localhost
      database: reserving
      trusted_connection: true
      encrypt: false
      trust_server_certificate: true
      timeout_seconds: 30
    claims:
      source: sql
      query_file: examples/sql/claims_template.sql
      params: [motor, 2018, 2024]
    premium:
      source: sql
      query_file: examples/sql/premium_template.sql
      params: [motor, 2018, 2024]
```

## Canonical field expectations

### Claims expected columns

- `id`
- `uw_year`
- `period`
- `paid`
- `outstanding`

### Premium expected columns

One of:

- `UnderwritingYear`, `Premium`
- `origin`, `development`, `Premium_selected`
- `uw_year`, `period`, `Premium_selected`

Preferred premium shape:

- If possible, use one row per UWY with `Premium_selected` at the selected/latest valuation.
- If using multiple development periods, ensure valuation semantics are consistent and validate latest diagonal premium values in Results.

## Practical advice

- Keep one segment per analytical context (line of business, portfolio, etc.).
- Use dedicated `session.path` per segment to avoid assumption overlap.
- Start with `quarterly` granularity, then evaluate `yearly` as a sensitivity check.
