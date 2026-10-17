# A4 throughput runs (PLAN.md A4.3, A6.8)

YOLOX-s @640x640 at detection intervals k = 1, 2 and 3. Each run is the synchronous pipeline, offline
(every frame processed, not paced in real time), on the first 150 frames of each MOT17 val scene, with
`configs/default.yaml` and its own scene corridor. There were 3 runs per k, one after another, on
2026-10-17 between 18:10 and 18:38 local time.
- `summary.json` is the median and range per k (`sightline runtime-summary`).
- The choice built from it is `../../detector_selection/a4_choice.json` (`sightline select-detector`).

- **Code:** the runs used a git worktree pinned at commit `12d15ea`, so later edits could not change a
  running measurement. These reports predate the `code` field that benchmark reports now carry, so
  the commit is recorded here.
- **Machine:** the demo laptop (AMD Family 25 Model 80, 16 logical CPUs, Windows 11), on AC power with
  the battery at 22% and the Windows "Balanced" power plan.
- **Slower than 14 Oct.** These runs are slower than the single offline runs of 2026-10-14
  (`../interval/yolox_s_640x640_k*`: 6.03, 13.08 and 18.89 fps at k = 1, 2 and 3). The code path is the
  same, so the machine state differed. A4 uses these runs, as its rule says: medians of 3 runs, made
  for the choice.
- **Why only YOLOX-s.** It has the highest in-scope F1 of the A4 candidates. A pair of any other
  detector has a lower F1, so it cannot be chosen whatever its speed. The choice file confirms that no
  unmeasured pair could change it.
