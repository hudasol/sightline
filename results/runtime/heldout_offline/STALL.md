# The 549 s frame in offline runtime run 1

**What the saved runtime file shows.** `frozen_k3_run1/report.json` (= `results/runtime_report.json`):
- MOT17-09 has `latency_ms_max` 548,939 ms and a wall time of 629.3 s for 525 frames, against a p50 latency of 11.9 ms.
- In an offline run, a frame's latency is measured inside its processing call. So one frame's processing took
  549 s.
- The project saves no per-frame log for benchmark runs, so the file cannot say which frame it was.

**What the Windows system log shows** (`windows_power_events_2026-10-17.txt` in this folder, exported read-only
on 2026-10-18):
- The run started at 20:27:24 local time with MOT17-09 (`post_tune` log line "start 20:27:24 at 4687efa").
- **20:27:39:** Kernel-Power event 105, "Power source change". Its first recorded value is `False`. As far as we
  know that field is the AC-online flag, which would mean the laptop went from the charger to battery. The log
  line itself does not name the field, so this reading is not verified.
- **20:27:39-20:27:50:** the system entered and left Modern Standby several times.
- **20:27:50:** the system entered Modern Standby.
- **20:37:02:** the system left it. That is **552 s in standby**, which covers the 549 s frame.
  MOT17-09 took 629.3 s in total. Without the 552 s in standby that leaves about 77 s for its 525 frames, against
  34.1 s and 36.1 s for the same clip in runs 2 and 3, so the rest of the clip also ran more slowly than usual.
  Why is not in the logs.

**Conclusion.** The stall coincides with 552 s of Modern Standby that began 26 s after the run's start line, right after a power
event. That is consistent with the process being suspended by standby rather than being slow. The log does not say why standby started: a closed lid, the screen
timing out on battery, or a manual sleep are all possible, and nothing distinguishes them.

**What was not done.** The held-out evaluation was not rerun, and the recorded bar check (3.14 fps, fail) is
unchanged. Runs 2 and 3 in this folder (14.79 and 14.87 fps) are separate timing checks of the same frozen system,
not a replacement for run 1.
