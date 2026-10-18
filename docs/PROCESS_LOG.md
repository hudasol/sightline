# Process log

Honest running record. Add entries as work happens; do not reconstruct afterwards.

| Date | What | Decision / evidence |
|---|---|---|
| 2026-10-09 | Repo created, PLAN.md committed before code | Plan fixed event definition, split rule, bars first |
| 2026-10-09 | README, then full pipeline + tests drafted with AI assistance | AI-assisted; all code to be read and understood by Huda before the mentor check-in |
| 2026-10-12 | MOT17 downloaded, 7 FRCNN train scenes extracted, split val 02/04/05/11/13, test 09/10; draft corridors; real detector benchmarked on val (2.76-6.27 fps) | Commits 1bfadcc..3b8bbe0 (PR #1) |
| 2026-10-12 | Input resize, decode thread, torch threads; candidate timings; detect-eval on val; moving camera scenes scored on detection and tracking only; draft-lock guard; NumPy 2 IDF1 crash fixed | PRs #2, #3 |
| 2026-10-14 | PLAN amendments A1-A4 (own clip bar and in scope people, detection interval, time based settings, own clip selection rule); MOT17 ignore-region bug fixed; SSDLite and YOLOX backends; threshold grid widened; CVAT import, annotated frames | PRs #4-#8; amendment A5 records the changes made after seeing val data |
| 2026-10-14 | Footage helpers (draw-corridor, label-events, check-clip), recording guide, detection worker thread with before/after latency, unresolved-event strips, model and data card updates, adversarial review | Commits on main 9f603ff onward; `docs/REVIEW_2026-10-14.md` |
| 2026-10-15 | M1: UNRESOLVED scoring checked (matching ignores status; 3 of the 6 track lost events matched GT); false event strips with a fixed selection rule. M2: solo recording guide | PRs #9, #10. The proposed MOT17-04 polygon waits for Huda's approval and is not used |
| 2026-10-17 | No own footage would exist before the 18 Oct deadline. Imported IndoorCrowd (tracking config, gated, Huda's Hugging Face login) and made it the held out evaluation set, split by site with all four sites in test (PLAN A6) | Decided before any detector, tracker or event output on IndoorCrowd existed. PR #11 |
| 2026-10-17 | Frame-rate-matched val: about-5 fps copies of the val scenes, so tuning sees the test frame rate without touching test. Event label rule moved to seconds (no 25/30 fps label changes) | PLAN A6.3, A6.5 |
| 2026-10-17 | Review items deferred on 14 Oct done before locking: detection age health check (async mode), A4 in code (`select-detector`), leakage tests per key, commit recorded in benchmark reports, honest test name | `docs/REVIEW_2026-10-14.md`, addendum |
| 2026-10-17 | A4 applied on MOT17 val: 3 sync offline runs per k for YOLOX-s @640 gave medians 3.98 / 7.30 / 10.13 fps at k = 1 / 2 / 3, so k = 3 (`results/detector_selection/a4_choice.json`) | The runs were slower than the 14 Oct single runs (13.08 fps at k = 2); used as measured, not rerun in search of a faster result |
| 2026-10-17 | Held out manifest locked (FINAL, 23 entries, 14 clips) at fabd2d3, then tuned on val only in a worktree pinned at f07eb40 (99 min) | `results/heldout_manifest.json`, `results/frozen_config.yaml`, `results/tuning/` |
| 2026-10-17 | Held out runtime report, then the single held out evaluation, then ablation, calibration, error analysis and stress on the frozen system | `results/`; nothing was retuned after the evaluation |
| 2026-10-18 | The mentor confirmed the task required sourcing data, not filming it. PLAN amendment A8: the evaluation uses sourced data only (MOT17, IndoorCrowd); own footage optional | Docs only; no held out number or setting changed |
| | Own corridor clips: recording and annotation | not recorded; a later version adds them with a fresh lock (A6.1) | Optional (A8) |

The rows dated 2026-10-12 and 2026-10-14 were added on 2026-10-14 from the git history and commit messages, not written as the work happened. That work was done with AI assistance.

## AI-use note
Code and doc drafts were produced with AI assistance. Anything described as measured must come from a saved result file. The first drafts were written where the real detector could not run. Since 2026-10-12 the real detectors have run on MOT17 val (`results/`). The 2026-10-15 and 2026-10-17 work (milestones M1-M3, the lock, tuning and the held out evaluation) was also done with AI assistance at Huda's direction; the rows for it were written as the work happened. No own footage exists.
