# Drop Review Packet

## Executive Summary

Drop review returned 7 candidates. The recommended candidate is `drop_ay2002_age39` with class `reasonable_alternative`.

## Recommendation / Result

- Recommendation class: reasonable_alternative
- Candidate id: drop_ay2002_age39
- Summary: The top-ranked candidate improves the review, but caveats remain material.

## Key Evidence

- Baseline drop recommendations: 6 found; first items: ('2003', 9), ('2002', 21), ('2002', 39), ('2001', 60), ('2001', 12).
- Baseline movement diagnostics: 20 findings; top finding codes: PREMIUM_LATE_MOVEMENT_1995_30, PREMIUM_LATE_MOVEMENT_1996_30, PREMIUM_LATE_MOVEMENT_1997_30, PREMIUM_LATE_MOVEMENT_1998_30, PREMIUM_LATE_MOVEMENT_1999_24.
- Baseline late-emergence rows: 5 rows; first origins: 1995, 1996, 1997, 1998, 1999.
- Baseline LDF findings: 5 findings; examples: AY 1995 observed age-to-age 2.182 differs from selected LDF 3.599 at age 3-6. | AY 1995 observed age-to-age 2.021 differs from selected LDF 2.477 at age 6-9..

## Candidate Ranking

- 1. `drop_ay2002_age39`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 2002 age 39
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 4109.435098, "drop_pairs": [["2002", 39]], "governance_tier": "amber", "ibnr_delta": 0.0, "ldf_finding_count": 29}`
- 2. `drop_ay2001_age60`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 2001 age 60
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 4109.435098, "drop_pairs": [["2001", 60]], "governance_tier": "amber", "ibnr_delta": 0.0, "ldf_finding_count": 29}`
- 3. `drop_ay2001_age12`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 2001 age 12
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 4022.796833, "drop_pairs": [["2001", 12]], "governance_tier": "amber", "ibnr_delta": -86.638265, "ldf_finding_count": 30}`
- 4. `drop_ay2002_age21`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 2002 age 21
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 3958.852236, "drop_pairs": [["2002", 21]], "governance_tier": "amber", "ibnr_delta": -150.582862, "ldf_finding_count": 28}`
- 5. `drop_ay1996_age87`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 1996 age 87
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 4109.435098, "drop_pairs": [["1996", 87]], "governance_tier": "amber", "ibnr_delta": 0.0, "ldf_finding_count": 29}`
- 6. `drop_ay2003_age9`
    - recommendation_class: reasonable_alternative
    - summary: Add drop for AY 2003 age 9
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 3784.999011, "drop_pairs": [["2003", 9]], "governance_tier": "amber", "ibnr_delta": -324.436087, "ldf_finding_count": 28}`
- 7. `drop_ay2002_age21__ay2003_age9`
    - recommendation_class: reasonable_alternative
    - summary: Combine the top two supported drop candidates
    - metrics: `{"baseline_total_ibnr": 4109.435098, "candidate_total_ibnr": 3651.212544, "drop_pairs": [["2003", 9], ["2002", 21]], "governance_tier": "amber", "ibnr_delta": -458.222554, "ldf_finding_count": 27}`

## A2A Factor Triangle

The table below shows the observed age-to-age (A2A) link ratios used in the LDF fitting. Empty cells indicate missing/NaN link ratios. Bolted drops are marked with ~~strikethrough~~.

| Origin | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 | 33 | 36 | 39 | 42 | 45 | 48 | 51 | 54 | 57 | 60 | 63 | 66 | 69 | 72 | 75 | 78 | 81 | 84 | 87 | 90 | 93 | 96 | 99 | 102 | 105 | 108 | 111 | 114 | 117 | 120 | 123 | 126 | 129 | 132 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1995 | 2.182 | 2.021 | 2.165 | 1.479 | 1.151 | 1.046 | 1.211 | 1.049 | 1.024 | 1.025 | 1.033 | 0.990 | 1.015 | 1.019 | 1.016 | 0.998 | 0.982 | 1.010 | 1.008 | 0.999 | 1.007 | 1.005 | 1.006 | 1.001 | 1.002 | 0.999 | 1.004 | 1.000 | 1.001 | 1.003 | 0.998 | 1.001 | 1.000 | 1.004 | 0.997 | 1.001 | 1.004 | 0.997 | 0.998 | 1.002 | 1.000 | 0.998 | 1.003 | 0.999 |
| 1996 | 3.238 | 1.485 | 1.807 | 1.482 | 1.203 | 1.255 | 1.209 | 1.065 | 1.067 | 1.015 | 1.030 | 0.997 | 1.004 | 1.026 | 1.017 | 1.011 | 0.992 | 1.009 | 1.012 | 0.996 | 1.010 | 1.002 | 1.002 | 0.999 | 0.998 | 1.012 | 0.996 | 0.997 | ~~1.008~~ | 1.002 | 1.014 | 0.998 | 1.005 | 1.001 | 0.998 | 1.000 | 1.002 | 0.998 | 1.002 | 0.998 |  |  |  |  |
| 1997 | 2.529 | 3.140 | 2.815 | 1.395 | 1.347 | 1.139 | 1.162 | 1.022 | 1.043 | 1.020 | 1.040 | 0.995 | 1.008 | 1.012 | 1.024 | 0.989 | 1.010 | 1.004 | 1.012 | 1.000 | 1.012 | 1.025 | 0.991 | 0.997 | 1.010 | 1.009 | 1.014 | 1.002 | 1.002 | 0.999 | 1.001 | 1.003 | 1.000 | 0.999 | 1.003 | 0.998 |  |  |  |  |  |  |  |  |
| 1998 | 4.300 | 2.488 | 2.224 | 1.651 | 1.461 | 1.275 | 1.221 | 1.046 | 1.034 | 1.054 | 1.018 | 1.024 | 1.016 | 1.019 | 1.035 | 0.989 | 1.033 | 1.030 | 1.008 | 1.002 | 1.005 | 1.001 | 1.020 | 1.002 | 1.013 | 1.017 | 1.002 | 1.002 | 1.002 | 1.002 | 1.004 | 1.001 |  |  |  |  |  |  |  |  |  |  |  |  |
| 1999 | 3.154 | 2.659 | 2.807 | 1.572 | 1.366 | 1.250 | 1.227 | 1.014 | 1.118 | 1.026 | 1.067 | 1.014 | 1.017 | 1.054 | 1.031 | 1.000 | 1.006 | 1.020 | 1.022 | 1.005 | 1.008 | 1.024 | 1.017 | 1.000 | 1.007 | 1.003 | 1.012 | 0.999 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2000 | 14.500 | 3.034 | 2.886 | 1.496 | 1.318 | 1.228 | 1.195 | 1.072 | 1.069 | 1.083 | 1.003 | 1.042 | 1.010 | 1.015 | 1.013 | 1.011 | 1.004 | 1.008 | 1.016 | 1.001 | 1.025 | 1.000 | 1.009 | 1.000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2001 | 6.250 | 6.040 | 2.205 | ~~2.333~~ | 0.853 | 1.291 | 1.154 | 1.076 | 1.098 | 1.027 | 1.036 | 1.052 | 0.998 | 1.033 | 1.010 | 1.000 | 1.010 | 1.000 | 1.005 | ~~1.020~~ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2002 | 17.000 | 3.382 | 2.522 | 1.628 | 1.714 | 1.303 | ~~1.464~~ | 1.048 | 0.931 | 1.062 | 1.060 | 1.073 | ~~0.944~~ | 1.060 | 1.011 | 0.990 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2003 | 6.333 | 4.737 | ~~7.689~~ | 0.863 | 1.556 | 0.950 | 1.265 | 0.978 | 1.077 | 1.019 | 1.027 | 0.993 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2004 | 9.500 | 3.632 | 2.688 | 1.571 | 1.297 | 1.193 | 1.232 | 1.091 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2005 | 3.762 | 1.456 | 2.600 | 1.411 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Actuarial Interpretation

The top-ranked candidate improves the review, but caveats remain material.

## Caveats

- No caveats returned by the deterministic review.

---

## Execution Details

### Command

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 10 --output harness/artifacts/drop_review_quarterly_20260514T175852Z.md
```

### Requested Inputs

- config_path: examples/config_quarterly.yml
- candidate_limit: 10

### Effective Inputs

- segment: quarterly
- granularity: quarterly
- dataset: quarterly
- quarterly_premium_csv: data/quarterly_premium.csv
- candidate_limit: 10

### Tool Calls

- `ConfigManager.from_yaml`: loaded reserving config.
inge t I want to have it more genera         `load_inputs_from_config`: loaded configured claims and premium dat that I for exampe can a.
 create inrred triang<tab>e
 
- `build_workflow_from_dataframes`: built deterministic `Reserving` workflow.
- `AssumptionReviewService.review_drops`: ran deterministic drop review.

### Data Lineage

- Config path: examples/config_quarterly.yml
- Dataset: quarterly
- Premium CSV: data/quarterly_premium.csv
- Segment: quarterly
- Granularity: quarterly

### Evidence References

- Evidence summary included in Key Evidence section (4 groups).
- Run metadata: `{"data_fingerprint": "dab17b69fb9050bbda15924cbd32a44aafc664e86c99c6e235843b7bb9ad1a60", "diagnostics_version": "v2.2", "generated_at": "2026-05-14T17:58:52.440415Z", "run_id": "41dc0392-f57b-4c19-aa32-7294a4b2ef57", "scenario_generator_version": "v1.2"}`

### Warnings

- No warnings captured.

### Reproducibility

- Timestamp: 2026-05-14T17:58:53.907836+00:00
- Code version: 26fd05f
- Review type: drop_review
- Candidate count: 7



 for the naming       
  it    
        dn't be j              
