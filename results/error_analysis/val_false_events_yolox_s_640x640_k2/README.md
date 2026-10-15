# False events on val: YOLOX-s @640, detect_every 2

Made with:

```bash
python -m sightline errors --false-events 10 --manifest configs/manifests/val.json --config configs/default.yaml --detector yolox_s --input-size 640 640 --detect-every 2 --results outputs/val_false_events_yolox_s_k2
```

This is the same run as `../val_unresolved_yolox_s_640x640_k2/`: same cached detections, same untuned default config, run synchronously. The per-clip details it rebuilt are byte-identical to that run's. Only the event clips MOT17-02 and MOT17-04 are included.

- `INDEX.md` and `index.json` are the selection: the rule, the run, the 6 events that met the rule and their frames. A false event is a predicted event that the event scorer left unmatched. The "Why it happened" column is left for a human. **No setting was changed from these strips.**
- Each strip runs from 10 frames before the track was first inside to the frame the event closed.
- For each false event, the index lists the GT events whose inside span overlaps it. In all 6, every overlapping GT event had already been matched to another predicted event. Matching is one-to-one, so this event was left over. The index does not say whether the track is a second track on an already-matched person or a track on something that is not a person; that needs a human to look at the strip.
- Three of the six (MOT17-02 events 5, 7 and 12) are also UNRESOLVED events in the unresolved index.
- The **strip images are not in git.** They are real MOT17 frames showing faces, MOT17 is under a non-commercial licence, and the project publishes no faces. They are in `outputs/val_false_events_yolox_s_k2/error_analysis/false_events/` (git-ignored); the command above recreates them.
