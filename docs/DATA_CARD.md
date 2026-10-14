# Data card

**Status: MOT17 downloaded and split (2026-10-12); own corridor clips not yet recorded or annotated.** Fill in the bracketed fields as clips are added (`configs/manifests/*.json` is the source of truth).

## MOT17 acquisition (2026-10-12)
- Downloaded `https://motchallenge.net/data/MOT17.zip`, 5,860,214,001 bytes (matches the server's Content-Length; server Last-Modified 18 Dec 2020). Every extracted member passed the zip CRC check.
- Extracted only `train/MOT17-{02,04,05,09,10,11,13}-FRCNN/{img1,gt,seqinfo.ini}` into `data/mot17/`; the DPM/SDP copies, public detections and test sequences were not kept, and the zip was deleted.
- **Licence:** the archive contains no licence or readme file, and the MOTChallenge site is now a static archive ("currently offline") that does not show licence text on the home or MOT17 pages. The CC BY-NC-SA 3.0 listing below is the commonly cited licence and is **not re-verified from a primary source**. Treat it as non-commercial, research use only.
- Corridors: `configs/corridors/MOT17-XX.yaml`, drafted from the first frame and ground-truth foot traces only, before any system output existed. Label-derived event counts: `results/labels/mot17_gt_events_draft.json`.

## Sources
| Source | Use | Licence | Notes |
|---|---|---|---|
| MOT17 train scenes (02, 04, 05, 09, 10, 11, 13) | public labelled pedestrian video; tuning and some held-out | CC BY-NC-SA 3.0 (commonly cited; no licence text found in the archive or on the archived site, 2026-10-12) | One detector copy per scene only. DPM/FRCNN/SDP copies are the same video and count as ONE scene. Non-commercial: do not redistribute inside a commercial product. |
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
