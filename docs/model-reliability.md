# Model reliability milestone

## Scope

Free professional single-game prediction app; all supplied leagues with a filter; pre-draft odds followed by completed-draft estimates. Model comparisons and charts appear in a compact, initially collapsed panel below predictions. Stop at this milestone for review.

## Reproducible experiment

Source: 2025_LoL_esports_match_data_from_OraclesElixir.csv. Valid games: 10044; complete drafts: 10034.

Source SHA-256: 882245414e94e13a9c70f14ed1047a36bffe2279f29526bffe884130d4aaf592

ML source SHA-256: 6451c71b04c167b02432a2c5742d111fa20813613ecb76d7d22677c4c33c2fb0

Seed: 42. Versions: {"scikit_learn": "1.9.1", "pandas": "3.0.6", "numpy": "2.5.4"}. Run: 2026-10-15T06:00:13.747577+00:00.

Command: python -m ml.train

| Window | Games | Start UTC | End UTC |
| --- | ---: | --- | --- |
| train | 6023 | 2025-01-11T11:11:24+00:00 | 2025-06-15T23:43:18+00:00 |
| calibration | 1004 | 2025-06-16T09:06:52+00:00 | 2025-07-29T23:25:57+00:00 |
| validation | 1472 | 2025-07-30T05:05:54+00:00 | 2025-09-01T23:04:22+00:00 |
| test | 1545 | 2025-09-02T05:05:49+00:00 | 2025-12-29T10:52:27+00:00 |

Chronological UTC-day split: 60% base training, 10% disjoint sigmoid calibration, 15% validation/model selection, 15% test. Boundary labels unavailable at the next window's start are purged. Only finished games update history; weights stay frozen through validation/test.

Lowest validation log loss per phase among Logistic Regression / Random Forest, raw / sigmoid calibrated. Classification threshold fixed at 0.5.

This test period was inspected for the earlier baseline. These comparisons are retrospective, not a new independent external test.

Raw and sigmoid-calibrated Logistic Regression and Random Forest use identical cohorts within each phase. Neither calibrated candidate was selected. Calibration is an evaluated option, not an assumed improvement.

## Selected test results

| Phase / model | Games | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_draft / logistic_regression | 1545 | 58.90% | 0.6190 | 0.6779 | 0.2415 | 0.0524 |
| post_draft / random_forest | 1535 | 58.31% | 0.6222 | 0.6719 | 0.2392 | 0.0416 |

On the same 1535 complete-draft games, pre-draft log loss is 0.6770, versus 0.6719 after draft; accuracy is 59.02% versus 58.31%. This does not establish a statistically significant gain or isolate patch features from other model changes.

All candidate scores, model parameters, plot data, and per-league pre-draft scores are in artifacts/evaluation.json. Auditable individual predictions are in artifacts/holdout_predictions.csv.

## Feature and serving boundaries

- Team Elo, prior win rate, recent form, experience, team patch results, and smoothed champion-role patch rates use only earlier completed games.
- Equal start timestamps share the same available history. In-progress games do not update histories.
- Duration schedules result availability; it is excluded from the model feature allowlist along with all current-game outcome statistics.
- Role/champion and champion-pair indicators capture associations, not causal composition strength.
- Patch strings remain as recorded, including 15.01 and 15.10.
- Serving model types are selected on validation and refitted separately. Calibrated serving candidates preserve a later disjoint calibration interval.
- Sandbox histories include all completed source games; the latest recorded patch is used for hypothetical matchups. Sparse patch history uses smoothed priors and visible notices. This is not a historical replay or a current-roster forecast.
- EDA losses are games minus wins. Full-dataset descriptive EDA does not feed features for earlier games.

## Verification

- 19 regression checks pass, covering duration availability, patch changes, patch string preservation, training/serving patch lookup parity, partition-boundary labels, and metric edge cases.
- Repeated training selected identical model types and reproduced selected test metrics within 1e-12. ML source fingerprint matches the saved report.
- Frontend TypeScript/Vite build and ESLint pass.
- Server rendering confirms a collapsed performance panel, both model names, valid values, and three accessible SVG charts.
- Real HTTP checks through the frontend proxy cover catalog, evaluation, both prediction phases, selected-model metadata, plot counts, and six invalid-input cases.
- Browser automation has no connected browser. Visual layout and browser interactions are not verified.

## Next proposed milestone

Data refresh and retraining workflow, with new-season validation and an independent forward evaluation. Current-season data, hosting, and scheduled retraining remain future work; review this milestone first.
