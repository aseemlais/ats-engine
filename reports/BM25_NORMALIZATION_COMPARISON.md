# BM25 normalisation comparison (generated)

Reference corpus: 64 documents (reliability: ok). Query method: `structured`, 28 tokens (24 unique).

## 1. Strong fresher match vs the reference population

| method | fresher raw | fresher normalised | median of other resumes | max of other resumes | gap (fresher - best other) |
|---|---|---|---|---|---|
| theoretical_bound | 47.03 | 0.302 | 0.070 | 0.287 | +0.016 |
| reference_percentile | 47.03 | 1.000 | 0.500 | 1.000 | +0.000 |
| legacy_ceiling | 44.27 | 0.503 | 0.123 | 0.478 | +0.026 |

## 2. Administrative JD text

| method | clean query | +admin via query builder (production) | +admin, builder bypassed (full text) |
|---|---|---|---|
| theoretical_bound | 0.302 | 0.302 | 0.177 |
| reference_percentile | 1.000 | 1.000 | 0.984 |
| legacy_ceiling | 0.503 | 0.503 | 0.194 |

## 3. Query length (first k unique query terms; fresher resume)

| k | theoretical_bound | reference_percentile | legacy_ceiling |
|---|---|---|---|
| 3 | 0.216 | 0.516 | 0.369 |
| 6 | 0.546 | 0.891 | 0.945 |
| 10 | 0.665 | 1.000 | 1.000 |
| 24 | 0.302 | 1.000 | 0.509 |

## 4. Repeated terms

| scenario | theoretical_bound | reference_percentile | legacy_ceiling |
|---|---|---|---|
| baseline | 0.302 | 1.000 | 0.503 |
| query repeated x5 | 0.302 | 1.000 | 0.347 |
| resume stuffed with python/sql/databricks x30 | 0.295 | 1.000 | 0.491 |

## 5. Unrelated documents

| document | theoretical_bound | reference_percentile | legacy_ceiling |
|---|---|---|---|
| ACCOUNTING | 0.000 | 0.000 | 0.000 |
| NURSING | 0.000 | 0.000 | 0.000 |
| GIBBERISH | 0.000 | 0.000 | 0.000 |