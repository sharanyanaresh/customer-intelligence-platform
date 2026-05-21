# RAG Evaluation Report

This report evaluates retrieval quality, refusal behavior, latency, and approximate prompt-token pressure across standard, adversarial, and edge-case questions.

| test_id | category | pass/fail | retrieval_score | latency_ms | token_count | refused | reason |
|---|---|---|---:|---:|---:|---|---|
| T01 | standard | PASS | 0.5203 | 27.33 | 161 | False | matched keywords: credit, report, inaccurate |
| T02 | standard | PASS | 0.5345 | 1.39 | 120 | False | matched keywords: incorrect, report |
| T03 | standard | PASS | 0.3566 | 1.83 | 729 | False | matched keywords: debt, owed |
| T04 | standard | PASS | 0.2673 | 1.74 | 699 | False | matched keywords: payment |
| T05 | standard | PASS | 0.2750 | 1.38 | 530 | False | matched keywords: account |
| T06 | standard | PASS | 0.3381 | 1.31 | 247 | False | matched keywords: student, loan, servicer |
| T07 | standard | PASS | 0.3125 | 1.26 | 935 | False | matched keywords: statement, due |
| T08 | standard | PASS | 0.4529 | 1.13 | 163 | False | matched keywords: fraudulent, inquiries |
| T09 | standard | PASS | 0.4020 | 1.28 | 466 | False | matched keywords: transfer |
| T10 | standard | PASS | 0.3320 | 1.63 | 740 | False | matched keywords: checking, savings |
| T11 | adversarial | PASS | 0.0000 | 1.55 | 12 | True | refused as expected |
| T12 | adversarial | PASS | 0.0000 | 1.40 | 13 | True | refused as expected |
| T13 | adversarial | PASS | 0.0000 | 1.49 | 9 | True | refused as expected |
| T14 | adversarial | PASS | 0.0000 | 1.63 | 9 | True | refused as expected |
| T15 | adversarial | PASS | 0.0000 | 1.90 | 11 | True | refused as expected |
| T16 | edge | PASS | 0.0000 | 1.83 | 1 | True | refused as expected |
| T17 | edge | PASS | 0.2318 | 1.65 | 508 | False | matched keywords: credit, report |
| T18 | edge | PASS | 0.0000 | 2.07 | 6 | True | refused as expected |
| T19 | edge | PASS | 0.0000 | 1.91 | 9 | True | refused as expected |
| T20 | edge | PASS | 0.0000 | 1.85 | 5 | True | refused as expected |

## Failure Analysis

No failures.
