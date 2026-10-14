# Unresolved events on val: YOLOX-s @640, detect_every 2

Made with:

```bash
python -m sightline errors --unresolved 10 --manifest configs/manifests/val.json --config configs/default.yaml --detector yolox_s --input-size 640 640 --detect-every 2 --results outputs/val_unresolved_yolox_s_k2
```

This used cached detections and the untuned default config, and ran synchronously. Only the event clips MOT17-02 and MOT17-04 are included.

- `INDEX.md` and `index.json` are the selection: the rule, the run, the 6 events that met the rule and their frames. The "Why it happened" column is left for a human. **No setting was changed from these strips.**
- The **strip images are not in git.** They are real MOT17 frames showing faces, MOT17 is under a non-commercial licence, and the project publishes no faces. They are in `outputs/val_unresolved_yolox_s_k2/error_analysis/unresolved/` (git-ignored); the command above recreates them.
