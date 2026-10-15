# Local 2025 retraining milestone

This milestone keeps the existing 2025 Oracle's Elixir CSV. It adds no remote data source, scheduled job, hosting service, or paid dependency.

## Workflow

Run `python -m ml.retrain` from the repository root using the project's virtual environment.

1. Copy the source into a temporary local snapshot and validate its structure, 2025 dates and patches, game/side/role uniqueness, consistent durations, matching player/team results, and one winner per game.
2. Preserve incomplete drafts for pre-draft analysis, report missing team IDs, and hash the snapshot. Existing matching snapshots are reused.
3. Compare the dataset and ML pipeline fingerprint with the active completed run. Identical inputs skip fitting unless `--force` is set.
4. Train into a new `artifacts/runs/<run-id>/` directory. The active run and original source stay intact during fitting.
5. Verify the model schema, catalog, finite evaluation results, saved-model probability outputs, and source checksum.
6. Write a completed run manifest and atomically select it in `artifacts/current.json`. The API resolves this pointer and caches immutable model files by path/version.

Run metadata records validation results, source snapshot, pipeline fingerprint, model checksum, start/end timestamps, and completion/failure state. Evaluation and individual predictions stay alongside the model that produced them.

`--validate-only` does not snapshot, train, or activate. `--status` displays the selected run. `--activate RUN_ID` revalidates and selects a previous completed run. `--activate legacy` selects the original root-level artifacts. Failed or checksum-mismatched runs cannot be activated.

## Dataset

The supplied file contains 120,528 rows and 10,044 games across 45 competitions. There are 10,034 complete drafts and 10 incomplete drafts; 606 team records have missing IDs and use the existing team-name fallback during feature construction.

The workflow validates structural consistency. It does not establish that every upstream champion assignment or statistic is factually correct. Evaluation continues to use the disclosed historical split; this milestone does not create a new independent test season.

## Website changes

- Removed the arrow decorations from both prediction buttons.
- After successful completed-draft submission, focus moves to the estimate heading and the page scrolls there.
- Failed submissions keep the user at the form with the error. Editing picks retains the earlier pre-draft estimate and clears the stale draft estimate.
- Reduced-motion users receive an immediate scroll.

## Verification

29 regression checks cover validation, snapshot reuse, incomplete drafts, failed fitting, activation, checksum errors, path validation, skipping unchanged inputs, and cached API activation/rollback, alongside the existing temporal-leakage checks.

Additional milestone checks: real training from a snapshot, active API model changes without restarting, a second unchanged invocation skipping training, frontend build/lint, and prediction requests through the frontend proxy. Browser automation has no connected browser, so actual scrolling and visual layout still need browser review.

Next proposed checkpoint: browser/mobile polish and usability review of the existing 2025 prediction experience, followed by planning free public hosting.
