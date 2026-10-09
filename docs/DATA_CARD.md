# Data card

**Status: data not yet collected or annotated.** Nothing below describes data that exists yet; it is the contract the data must meet before any number is reported. Fill in the bracketed fields as clips are added (`manifests/*.json` is the source of truth).

## Sources
| Source | Use | Licence | Notes |
|---|---|---|---|
| MOT17 train scenes (02, 04, 05, 09, 10, 11, 13) | public labelled pedestrian video; tuning and some held-out | CC BY-NC-SA 3.0 (re-verify at download) | One detector copy per scene only. DPM/FRCNN/SDP copies are the same video and count as ONE scene. Non-commercial: do not redistribute inside a commercial product. |
| Self-recorded indoor corridor clips | the real target domain | own footage | Consent of anyone identifiable required; no faces published. |

## Split rule
Split by whole **scene / camera / recording session**, never by frame or by clip within a session. `sightline split-check` fails on any shared id, path, scene, base scene, camera or session. The held-out manifest is hashed (`lock-manifest`) **before** any tuning; `tune` refuses to run without the lock and `evaluate` refuses if hashes changed.

## Held-out coverage required (brief)
≥12 clips, including negatives (nobody enters) and each of: clear, occlusion/crowding, camera motion/blur, low light/compression. `coverage_check` reports gaps.

## Labels
- Person boxes: MOTChallenge format (`frame,id,x,y,w,h,conf,class,vis`). MOT17 labels as published; self-recorded clips annotated by [annotator, tool, date].
- Ground-truth **events**: for MOT17 derived mechanically from GT boxes with the same rule constants as the system (`configs/eval.yaml`, never tuned); for own clips hand-marked first-inside frames, reviewed [by whom].
- Known label limits: MOT17 GT is not a corridor annotation; derived events are a proxy. Own-clip event labels are single-annotator unless stated.

## Counts (fill in)
| Split | Clips | Frames | Minutes | Events | Negative minutes |
|---|---|---|---|---|---|
| val | [ ] | [ ] | [ ] | [ ] | [ ] |
| held-out | [ ] | [ ] | [ ] | [ ] | [ ] |

## Known biases / limitations
MOT17 is mostly outdoor street/mall footage with different camera heights than a corridor; indoor clips are from one or few locations and will not cover all lighting, clothing or crowd types. Results do not generalise beyond the conditions listed in the manifest.
