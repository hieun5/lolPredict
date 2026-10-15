# Desktop usability milestone

Scope: desktop web, the existing 2025 dataset, and the current free local prediction experience. No mobile layout or model changes are included.

## Changes

- Both prediction sections are native labeled forms supporting keyboard submission.
- Swap sides moves the selected teams and their champion drafts together, and clears the stale estimate.
- Draft legends name the selected team. Remaining picks are listed by side and role, with an accessible progress message explaining why the draft button is disabled.
- Each prediction phase has its own pending label, inline error, and focused error feedback. A 15-second abort prevents a stalled request from leaving the form disabled indefinitely.
- Performance details load and retry independently. A chart failure does not block prediction or reset team/champion selections.
- Successful completed-draft submissions retain the focus/scroll behavior, respect reduced motion, and display both drafts alongside the estimate.
- Edit teams and Edit draft links return to the corresponding labeled heading.

## Checks

Seven frontend interaction tests pass using Vitest, jsdom, and React Testing Library. Coverage includes performance failure/retry, swapping sides, duplicate champion prevention and progress, successful focus/scroll, reduced motion, inline API errors, preserving pre-draft results after edits, and timeout recovery.

Frontend build and ESLint pass. Real HTTP prediction and invalid-input checks through the frontend proxy pass. These tests do not establish visual browser layout correctness.

Native browser inspection was attempted using the computer-use skill. The tool rejected Chrome access with: "Computer Use was not approved to use Google Chrome." Visual Chrome inspection remains pending app approval.

## Review boundary

The previous prediction, model evaluation, and local retraining work was committed and pushed as `02eca26`. This desktop milestone receives its own subsequent commit. Continue with visual desktop review when access is available, then choose a free hosting setup in a separate milestone.
