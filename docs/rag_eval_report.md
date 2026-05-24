# RAG Evaluation Report

This report evaluates retrieval quality, refusal behavior, latency, and approximate prompt-token pressure across standard, adversarial, and edge-case questions.

| test_id | category | pass/fail | retrieval_score | latency_ms | token_count | refused | reason |
|---|---|---|---:|---:|---:|---|---|
| T01 | standard | PASS | 0.5203 | 48.65 | 161 | False | matched keywords: credit, report, inaccurate |
| T02 | standard | PASS | 0.5345 | 1.98 | 120 | False | matched keywords: incorrect, report |
| T03 | standard | PASS | 0.3566 | 1.38 | 729 | False | matched keywords: debt, owed |
| T04 | standard | PASS | 0.2673 | 1.49 | 699 | False | matched keywords: payment |
| T05 | standard | PASS | 0.2750 | 2.19 | 530 | False | matched keywords: account |
| T06 | standard | PASS | 0.3381 | 3.64 | 247 | False | matched keywords: student, loan, servicer |
| T07 | standard | PASS | 0.3125 | 3.45 | 935 | False | matched keywords: statement, due |
| T08 | standard | PASS | 0.4529 | 2.45 | 163 | False | matched keywords: fraudulent, inquiries |
| T09 | standard | PASS | 0.4020 | 2.71 | 466 | False | matched keywords: transfer |
| T10 | standard | PASS | 0.3320 | 3.11 | 740 | False | matched keywords: checking, savings |
| T11 | adversarial | PASS | 0.0000 | 3.32 | 12 | True | refused as expected |
| T12 | adversarial | PASS | 0.0000 | 5.86 | 13 | True | refused as expected |
| T13 | adversarial | PASS | 0.0000 | 2.88 | 9 | True | refused as expected |
| T14 | adversarial | PASS | 0.0000 | 2.41 | 9 | True | refused as expected |
| T15 | adversarial | PASS | 0.0000 | 2.29 | 11 | True | refused as expected |
| T16 | edge | PASS | 0.0000 | 1.43 | 1 | True | refused as expected |
| T17 | edge | PASS | 0.2318 | 1.63 | 508 | False | matched keywords: credit, report |
| T18 | edge | PASS | 0.0000 | 1.99 | 6 | True | refused as expected |
| T19 | edge | PASS | 0.0000 | 1.83 | 9 | True | refused as expected |
| T20 | edge | PASS | 0.0000 | 1.58 | 5 | True | refused as expected |

## Failure Analysis

No failures.
