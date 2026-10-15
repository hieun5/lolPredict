# Vercel preview milestone

## Deployment

The project `lolpropredictor` is linked to the user-confirmed `hieu-nguyen4` Hobby account. No paid integrations or plan changes were made.

- Preview: https://lolpropredictor-16qfbmnc5-hieu-nguyen4.vercel.app
- Dashboard: https://vercel.com/hieu-nguyen4/lolpropredictor/FKNFgoC1SMMefg4MF53KjanZwyLL
- Deployment ID: `dpl_FKNFgoC1SMMefg4MF53KjanZwyLL`
- Target: preview; status: Ready
- Build: 3 minutes 28 seconds on the default 2 vCPU / 8 GB build machine
- Runtime: Python 3.14; Node.js 24 for the frontend build
- Inspector-reported FastAPI function size: 61.86 MB

Deployment protection remains enabled. Open the preview while signed in to the owning Vercel account, or use the authenticated smoke command below. This preview milestone does not release the app on its production domain or connect automatic deployments to Git pushes.

## Cloud build result

Vercel rebuilt the frontend and trained/verified its own model from the tracked 2025 CSV. It reproduced the local test results:

| Model phase | Accuracy | ROC AUC | Test games |
| --- | --- | --- | --- |
| Pre-draft Logistic Regression | 0.588997 | 0.618964 | 1,545 |
| Completed-draft Random Forest | 0.583062 | 0.622152 | 1,535 |

The paired dataset contains 10,044 games, including 10,034 complete drafts. Its final result cutoff is `2025-12-29T11:22:32+00:00`.

- Source SHA256: `882245414e94e13a9c70f14ed1047a36bffe2279f29526bffe884130d4aaf592`
- Model SHA256: `1076a7fddbfe9c47dece0b554a26e889ca46fe5bb5931df790d28878d8620cde`
- Training code SHA256: `fbd611d9654d05dd75448c72f9f8e858b1dbec3a495d69934ea456b64877977d`
- Packages: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0

These remain retrospective historical results, not evidence of current-season performance. See [model reliability](model-reliability.md).

## Fixes found by cloud deployment

The first upload failed because Vercel's requirements parser rejected the UTF-8 byte-order mark at the beginning of `requirements.txt`. The file now uses UTF-8 without that marker; package pins are unchanged. Runtime application code is the prior hosting milestone plus this encoding fix. Verification scripts and this report were added after the deployed application files were uploaded.

Vercel automatically classified that first project deployment as production despite `--target preview`. It failed before serving the app. The subsequent successful deployment was explicitly inspected and confirmed as preview. Do not treat a deployment request's initial success message as evidence that its build has finished; verify the final target and Ready state.

The HTTP checker now supports authenticated `vercel curl` without printing or persisting credentials. It passes JSON through standard input, preserves application error status codes, and checks the compiled app assets while allowing Vercel's injected preview toolbar script. The pinned CLI forwards `--non-interactive` to system curl, so that flag is omitted for curl requests; a local project link is required before testing.

## Verification commands

```powershell
npx --yes vercel@64.2.0 project inspect --non-interactive
npx --yes vercel@64.2.0 inspect lolpropredictor-16qfbmnc5-hieu-nguyen4.vercel.app --scope hieu-nguyen4 --non-interactive
.\.cache\hosting-venv\Scripts\python.exe tests/smoke_api.py --base-url https://lolpropredictor-16qfbmnc5-hieu-nguyen4.vercel.app --hosted --vercel
.\.cache\hosting-venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Remaining release work

Production was launched in the subsequent [production milestone](production-release.md) at https://lolpropredictor.vercel.app. The following records the remaining work at the original preview checkpoint.

All 38 Python checks passed. Authenticated HTTP verification passed against the exact preview URL: page, compiled JavaScript/CSS/favicon, model readiness, catalog, evaluation and diagnostics, both prediction phases, six invalid inputs, unknown API routes, and five private-file paths. The equivalent local HTTP check also passed. The cloud sample matchup (BNK FEARX vs DN SOOPers) returned blue probabilities of 0.833605 before draft and 0.826745 with the smoke test's completed draft, matching local inference to numerical precision. A deployment-scoped runtime error query over the preceding 30 minutes returned zero records after these checks.

Complete the desktop browser review, then stage and verify a production build before assigning the public domain. Browser review is separate from HTTP verification: the latter cannot establish the appearance of native selects or actual focus/scroll behavior in a browser.

Browser approval denied the local preview origin during this milestone. The user was given the official Settings > Browser > Agent permissions steps and asked to confirm once access is enabled. No alternate browser or debugging path was used to work around that denial.

## References

- [Vercel curl for protected deployments](https://vercel.com/docs/cli/curl)
- [Vercel's first-deployment production behavior](https://vercel.com/blog/default-production-domain)
- [Staging production without assigning domains](https://vercel.com/docs/cli/deploy#skip-domain)
