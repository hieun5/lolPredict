# Production release

## Launch result

**Live site: https://lolpropredictor.vercel.app**

The production launch completed on October 15, 2026. Vercel reports `READY`, and the registered public domain resolves to the exact staged and tested production deployment. Anonymous HTTP checks passed without Vercel credentials or browser cookies.

The checks covered the page, compiled JavaScript/CSS/favicon, model readiness, catalog, evaluation/diagnostics, pre-draft predictions, completed-draft predictions, six invalid requests, unknown API routes, and private-file paths. The same full checks passed against the staged deployment using authenticated Vercel curl before promotion. This milestone changes release documentation; the application code remains the tested source commit below.

The production domain is `lolpropredictor.vercel.app`. Vercel's generated team alias `lolpropredictor-hieu-nguyen4.vercel.app` still redirects anonymous visitors to Vercel sign-in. The project domain list was used to identify the correct public URL. No deployment-protection settings were changed: standard protection (`all_except_custom_domains`) remains enabled for previews and generated deployment URLs.

## Release procedure

Production uses the confirmed `hieu-nguyen4/lolpropredictor` Hobby project, Python 3.14, Node.js 24, and the repository-root FastAPI configuration. No paid services or custom domain purchases are part of this release.

The staged deployment is `dpl_3vA2MmpmPTknQdLbB4doHJWwkzM8` at https://lolpropredictor-54tti4mt2-hieu-nguyen4.vercel.app. Vercel reported target `production`, state `READY`, a build duration of 3 minutes 23 seconds, and a FastAPI function size of 61.86 MB in `iad1`.

The serving model checksum recorded by the build is `ada0126a3f324f2a0901bbfbb63164636ef5b816bc422737424708347975b586`. Its source checksum is `882245414e94e13a9c70f14ed1047a36bffe2279f29526bffe884130d4aaf592`, and its training code checksum is `fbd611d9654d05dd75448c72f9f8e858b1dbec3a495d69934ea456b64877977d`. The build verified 10,044 games, including 10,034 complete drafts, with a final result cutoff of `2025-12-29T11:22:32+00:00`.

The application source for this launch is commit `835aa2bc02921952f3799c62aa047979c08809f2`. The release builds a model from the existing 2025 CSV, with the same chronological evaluation and model verification used in the preview milestone. Training happens during builds; requests use the resulting immutable release.

The deployment is staged with:

```powershell
npx --yes vercel@64.2.0 project inspect --non-interactive
npx --yes vercel@64.2.0 deploy --prod --skip-domain --scope hieu-nguyen4 --non-interactive --no-wait --json
```

The public domain is assigned only after inspecting the final Ready state and passing the authenticated hosted smoke checks against the staged production URL. Promotion uses that same production deployment; it does not rebuild the earlier preview.

```powershell
.\.cache\hosting-venv\Scripts\python.exe tests/smoke_api.py --base-url https://lolpropredictor-54tti4mt2-hieu-nguyen4.vercel.app --hosted --vercel
npx --yes vercel@64.2.0 promote dpl_3vA2MmpmPTknQdLbB4doHJWwkzM8 --scope hieu-nguyen4 --non-interactive --yes --timeout 50s
.\.cache\hosting-venv\Scripts\python.exe tests/smoke_api.py --base-url https://lolpropredictor.vercel.app --hosted
```

The final smoke command intentionally uses direct HTTP without Vercel credentials, proving that visitors can access the page and prediction endpoints without signing in. Preview protection settings stay unchanged.

## Operations

The `/api/health` endpoint verifies model readiness and reports the season, data cutoff, and selected models. Vercel's runtime logs are available for investigating errors. No external monitoring, log drain, load test, or cold-start benchmark is configured by this milestone.

After the anonymous production checks, a deployment-scoped `level:error` query over the preceding 30 minutes returned zero records. This is a release-time log check, not ongoing alerting.

Production remains a historical 2025 sandbox with all recorded leagues and a league filter. Its results are retrospective, and its team histories end at the dataset cutoff. It does not supply current rosters, schedules, live match data, or betting-market odds.

GitHub commits do not automatically deploy: the project is linked locally through the CLI, and Git integration has not been enabled. Keep credentials and `.vercel/` outside Git. For later releases, stage and test first, promote the tested deployment, and record its ID. A known working production deployment can be used as the rollback target for a subsequent release.

Desktop browser visual review is still separate from these automated HTTP checks. The local browser origin was denied in the preceding milestone, and no confirmation that browser access has been enabled has been received. The production launch does not claim visual validation.

The user's selected next milestone is polishing the existing prediction page. Keep that work focused on the desktop experience and stop at its next decision checkpoint.

## References

- [Vercel staged production deployments](https://vercel.com/docs/cli/deploy#skip-domain)
- [Promote an existing production deployment](https://vercel.com/docs/cli/promote)
- [Vercel runtime logs](https://vercel.com/docs/cli/logs)
