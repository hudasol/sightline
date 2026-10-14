# Process log

Honest running record. Add entries as work happens; do not reconstruct afterwards.

| Date | What | Decision / evidence |
|---|---|---|
| 2026-10-09 | Repo created, PLAN.md committed before code | Plan fixed event definition, split rule, bars first |
| 2026-10-09 | README, then full pipeline + tests written with Claude (Claude Code, Sonnet 5.5) | AI-assisted; all code to be read and understood by Huda before the mentor check-in |
| 2026-10-12 | MOT17 downloaded, 7 FRCNN train scenes extracted, split val 02/04/05/11/13, test 09/10; draft corridors; real detector benchmarked on val (2.76-6.27 fps) | Commits 1bfadcc..3b8bbe0 (PR #1) |
| 2026-10-12 | Input resize, decode thread, torch threads; candidate timings; detect-eval on val; moving-camera scenes scored on detection and tracking only; draft-lock guard; NumPy 2 IDF1 crash fixed | PRs #2, #3 |
| 2026-10-14 | PLAN amendments A1-A4 (own-clip bar and in-scope people, detection interval, time-based settings, own-clip selection rule); MOT17 ignore-region bug fixed; SSDLite and YOLOX backends; threshold grid widened; CVAT import, annotated frames | PRs #4-#8; amendment A5 records the changes made after seeing val data |
| 2026-10-14 | Footage helpers (draw-corridor, label-events, check-clip), recording guide, detection worker thread with before/after latency, unresolved-event strips, model and data card updates, adversarial review | Commits on main 9f603ff onward; `docs/REVIEW_2026-10-14.md` |
| | Own corridor clips: recording and annotation | [pending, needs Huda's footage] |
| | Detector and detection interval selection on own val clips (PLAN A4) | [pending] |
| | Final lock, tuning, held-out evaluation | [pending] |

The rows dated 2026-10-12 and 2026-10-14 were added on 2026-10-14 from the git history and commit messages, not written as the work happened. That work was done with Claude Code (Opus 5.5).

## AI-use note
Code and doc drafts were produced with Claude. Anything described as measured must come from a saved result file. The first drafts were written where the real detector could not run. Since 2026-10-12 the real detectors have run on MOT17 val (`results/`). No own footage exists yet.
