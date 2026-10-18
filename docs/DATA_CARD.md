# Data card

**Status (2026-10-17):** MOT17 (val and part of held-out) and IndoorCrowd (held-out, PLAN.md A6) are split and the held-out manifest is locked (`results/heldout_manifest.json`). Own corridor clips are not recorded; a later version adds them with a fresh lock. **The evaluation uses sourced data only (MOT17 and IndoorCrowd); no own footage was recorded or required (PLAN.md A8).** Own clips are optional. `configs/manifests/*.json` is the source of truth.

## MOT17 acquisition (2026-10-12)
- Downloaded `https://motchallenge.net/data/MOT17.zip`, 5,860,214,001 bytes (matches the server's Content-Length; server Last-Modified 18 Dec 2020). Every extracted member passed the zip CRC check.
- Extracted only `train/MOT17-{02,04,05,09,10,11,13}-FRCNN/{img1,gt,seqinfo.ini}` into `data/mot17/`; the DPM/SDP copies, public detections and test sequences were not kept, and the zip was deleted.
- **Licence:** the archive contains no licence or readme file, and the MOTChallenge site is now a static archive ("currently offline") that does not show licence text on the home or MOT17 pages. The CC BY-NC-SA 3.0 listing below is the commonly cited licence and is **not re-verified from a primary source**. Treat it as non-commercial, research use only.
- Corridors: `configs/corridors/MOT17-XX.yaml`, drafted from the first frame and ground-truth foot traces only, before any system output existed. Label-derived event counts: `results/labels/mot17_gt_events_draft.json`.

## IndoorCrowd acquisition (2026-10-17)
- **What:** the `tracking` config only of `sebnae/IndoorCrowd` (Hugging Face, gated), commit `87049647f27c4419dcba051474dcfc0fcecad8e0`. Four parquet files, one per site, downloaded with the Hugging Face CLI under Huda's account after accepting the gate. No detection/segmentation config was downloaded.
- **Converted** with `python -m sightline import-indoorcrowd` into `data/indoorcrowd/<site>/<recording>/` (MOT format). File hashes, frame counts and box counts: `results/ingest/indoorcrowd_import.json`. 12 recordings, 1,305 frames, 9,837 boxes, 1280x720, 5 fps.
- **Recordings are distinct videos.** Two recordings of one site can share frame numbers (both start at `frame_00001`), so this was checked before splitting: frames with the same number differ by 6 to 63 grey levels on average (64x36 thumbnails), so no recording repeats another's frames. Frame numbers within each recording are contiguous.
- **Cameras are fixed** (project page: "Fixed webcam at 1280 x 720 px"), so an image-space corridor stays on the floor within a recording.
- **Correction (2026-10-17, after the evaluation, PLAN.md A7.1):** ACS-EC has **two** camera views, not one. `clip_00001_00051` is an elevated view of the atrium; `clip_00001_00082` and `clip_00026_00091` are a ground-level view of the lounge. The ACS-EC corridor was drawn on the first view, so on the other two recordings it covers the counter area and its negative corridor lies mostly on a wall. The other sites have one view each.
- **Visibility is not given.** Boxes are written with visibility -1 (unknown), never a made-up 1.0. The in-scope rule judges them by height alone (A1 and A6 in PLAN.md).
- **Labels** were made by the dataset authors: foundation-model auto-labelling, then human correction of false positives, missing boxes and identity switches, with linear interpolation of gaps (dataset card). Interpolated boxes are not marked, so a box can be interpolated rather than seen.
- **Faces** are blurred in the source by the dataset authors. Sightline still publishes no IndoorCrowd frame.
- **Licence:** `docs/licences/INDOORCROWD_LICENSE.txt`. The dataset declares `fair-noncommercial-research-license` and its card says it is for non-commercial computer vision research only and prohibits surveillance and any re-identification of individuals. Sightline applies the **stricter reading** of the card and the licence text together: offline evaluation only, never a live feed, monitoring or counting of real people, no re-identification (anonymous per-clip ids from geometry only), no redistribution of frames or labels, no commercial use of any result.
- **Open question for the project owner:** whether this task (an internship training task) counts as "non-commercial computer vision research" for IndoorCrowd and MOT17 is for Huda and the mentor to confirm. It is not decided here.

## Sources
| Source | Use | Licence | Notes |
|---|---|---|---|
| MOT17 train scenes (02, 04, 05, 09, 10, 11, 13) | public labelled pedestrian video; tuning and some held-out | CC BY-NC-SA 3.0 (commonly cited; no licence text found in the archive or on the archived site, 2026-10-12) | One detector copy per scene only. DPM/FRCNN/SDP copies are the same video and count as ONE scene. Non-commercial: do not redistribute inside a commercial product. |
| IndoorCrowd `tracking` config (sites ACS-EC, ACS-EG, IE-Central, R-Central) | indoor, fixed-camera labelled tracking video: the primary held-out evaluation set (PLAN.md A6) | fair-noncommercial-research-license (`docs/licences/INDOORCROWD_LICENSE.txt`); non-commercial research only, no surveillance, no re-identification | Gated. Split by site, never by frame or recording. Faces blurred by the authors; no frame published. |
| Self-recorded indoor corridor clips | the real target domain. **Optional; not recorded and not used in the evaluation (PLAN.md A8)** | own footage | Consent of anyone identifiable required; no faces published. |

## Split rule
Split by whole **scene / camera / recording session**, never by frame or by clip within a session. `sightline split-check` fails on any shared id, path, scene, base scene, camera or session. The held-out manifest is hashed (`lock-manifest`) **before** any tuning; `tune` refuses to run without the lock and `evaluate` refuses if hashes changed.

## Held-out coverage required (brief)
>= 12 clips, including negatives (nobody enters) and each of: clear, occlusion/crowding, camera motion/blur, low light/compression. `coverage_check` reports gaps, including which difficult conditions the **event** metrics rest on: moving-camera clips count for detection and tracking only.

## Labels
- Person boxes: MOTChallenge format (`frame,id,x,y,w,h,conf,class,vis`). MOT17 labels as published; self-recorded clips annotated by [annotator, tool, date].
- Ground-truth **events**: for MOT17 derived mechanically from GT boxes and the corridor by a fixed label rule (`configs/eval.yaml`: at least 3 frames inside, gaps of up to 5 frames merged; never tuned). That rule is separate from the system's own tunable event settings in `configs/default.yaml`; for own clips hand-marked first-inside frames, reviewed [by whom].
- Known label limits: MOT17 GT is not a corridor annotation; derived events are a proxy. Own-clip event labels are single-annotator unless stated.
- **MOT17 classes (checked 2026-10-14):** class 1 conf 1 is scored. Classes 2, 7, 8 and 12, and conf-0 pedestrians, are ignore regions. All other classes are dropped. See PLAN.md amendment A1.

## Who is "in scope" (PLAN.md amendment A1, fixed 2026-10-14)
A scored pedestrian is **in scope** when both hold:
- box height >= **10% of the source frame height**: at least 108 px at 1080p, 48 px at 480p, 36 px at 360p
- visibility (column 9) >= **0.65**

The thresholds live in `configs/eval.yaml` under `in_scope`; that file is hashed into the held-out lock. They follow the Caltech Pedestrian "Reasonable" subset (>= 50 px of a 480 px frame, >= 65% visible), expressed relative to frame height. They were fixed before any in-scope number was computed.

- **Ignore, not penalise.** For in-scope scoring, an out-of-scope person becomes an **ignore region**. A detection on them is neither a hit nor a false positive, and missing them is not a miss.
- **Two reports.** Detection and tracking are reported **in scope** and on **all boxes**.
- **The bar.** The detection bar is judged on held-out own clips, in scope. MOT17 is reported, not judged. (Superseded: with no own clips it is judged on held-out IndoorCrowd, in scope; PLAN.md A6.1, A8.)
- **Events** use every scored person, whatever their scope. A person inside the corridor counts even if small or occluded.

## Own-clip annotation rules (PLAN.md amendment A1; optional, A8)
- **Every person box records visibility** (0-1, the visible fraction of the person) in column 9. The in-scope test (height >= 10% of the frame and visibility >= 0.65) needs it. A box with no real visibility estimate must not be exported as 1.0.
- Label every person in view, in scope or not. Out-of-scope people become ignore regions, so a detection on them is never counted as a false positive.
- Record which frames were annotated. Unannotated frames are not scored, and empty annotated frames are real negatives.

## Counts (computed 2026-10-17 from the manifests and ground truth, not typed)
| Split | Distinct clips | Frames | Minutes | GT events | Negative entries | Negative minutes |
|---|---|---|---|---|---|---|
| val | 10 entries: 5 MOT17 scenes and their 5 about-5 fps copies | 4,991 | 5.83 | 41 (02: 11, 04: 9, 02-5fps: 12, 04-5fps: 9) | 0 | 0 |
| held-out | 14 (MOT17-09, MOT17-10, 12 IndoorCrowd) | 2,484 | 5.00 | 80 (MOT17-09: 14, IndoorCrowd: 66) | 9 | 3.63 |

GT events are counted on entries scored on events, using the label rule in seconds (A6.5). The 5 fps copy of
MOT17-02 has 12 events where the original has 11. The copy keeps every 6th frame and the rule is then 1 frame
inside with a 1-frame merge gap, so derived events need not match one for one; which event splits or appears
was not examined.

Per entry:

| Split | Entry | Dataset | fps | Frames | Seconds | GT boxes | GT events | Scored on | Negative | Conditions |
|---|---|---|---|---|---|---|---|---|---|---|
| val | MOT17-02-FRCNN | mot17 | 30.0 | 600 | 20.0 | 18581 | 11 | detection,tracking,events |  | occlusion_crowding |
| val | MOT17-04-FRCNN | mot17 | 30.0 | 1050 | 35.0 | 47557 | 9 | detection,tracking,events |  | occlusion_crowding,low_light_compression |
| val | MOT17-05-FRCNN | mot17 | 14.0 | 837 | 59.8 | 6917 | - | detection,tracking |  | camera_motion_blur |
| val | MOT17-11-FRCNN | mot17 | 30.0 | 900 | 30.0 | 9436 | - | detection,tracking |  | camera_motion_blur |
| val | MOT17-13-FRCNN | mot17 | 25.0 | 750 | 30.0 | 11642 | - | detection,tracking |  | camera_motion_blur,occlusion_crowding |
| val | MOT17-02-FRCNN-5fps | mot17 | 5.0 | 100 | 20.0 | 3092 | 12 | tracking,events |  | occlusion_crowding |
| val | MOT17-04-FRCNN-5fps | mot17 | 5.0 | 175 | 35.0 | 7920 | 9 | tracking,events |  | occlusion_crowding,low_light_compression |
| val | MOT17-05-FRCNN-5fps | mot17 | 4.67 | 279 | 59.8 | 2302 | - | tracking |  | camera_motion_blur |
| val | MOT17-11-FRCNN-5fps | mot17 | 5.0 | 150 | 30.0 | 1571 | - | tracking |  | camera_motion_blur |
| val | MOT17-13-FRCNN-5fps | mot17 | 5.0 | 150 | 30.0 | 2335 | - | tracking |  | camera_motion_blur,occlusion_crowding |
| held-out | MOT17-09-FRCNN | mot17 | 30.0 | 525 | 17.5 | 5325 | 14 | detection,tracking,events |  | clear |
| held-out | MOT17-10-FRCNN | mot17 | 30.0 | 654 | 21.8 | 12839 | - | detection,tracking |  | camera_motion_blur,low_light_compression |
| held-out | IC-acs_ec-clip_00001_00051 | indoorcrowd | 5.0 | 51 | 10.2 | 222 | 3 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-acs_ec-clip_00001_00082 | indoorcrowd | 5.0 | 82 | 16.4 | 2191 | 11 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-acs_ec-clip_00026_00091 | indoorcrowd | 5.0 | 41 | 8.2 | 539 | 5 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-acs_eg-clip_00001_00061 | indoorcrowd | 5.0 | 61 | 12.2 | 282 | 4 | detection,tracking,events |  | clear |
| held-out | IC-acs_eg-clip_00011_00093 | indoorcrowd | 5.0 | 73 | 14.6 | 365 | 6 | detection,tracking,events |  | clear |
| held-out | IC-acs_eg-clip_00056_00191 | indoorcrowd | 5.0 | 81 | 16.2 | 587 | 11 | detection,tracking,events |  | clear |
| held-out | IC-ie_central-clip_00001_00061 | indoorcrowd | 5.0 | 61 | 12.2 | 356 | 3 | detection,tracking,events |  | clear |
| held-out | IC-ie_central-clip_00001_00094 | indoorcrowd | 5.0 | 94 | 18.8 | 687 | 2 | detection,tracking,events |  | clear |
| held-out | IC-ie_central-clip_00001_00381 | indoorcrowd | 5.0 | 381 | 76.2 | 2039 | 9 | detection,tracking,events |  | clear |
| held-out | IC-r_central-clip_00001_00090 | indoorcrowd | 5.0 | 90 | 18.0 | 550 | 3 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-r_central-clip_00001_00134 | indoorcrowd | 5.0 | 134 | 26.8 | 1147 | 5 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-r_central-clip_00001_00156 | indoorcrowd | 5.0 | 156 | 31.2 | 872 | 4 | detection,tracking,events |  | occlusion_crowding |
| held-out | IC-acs_ec-clip_00001_00051-neg | indoorcrowd | 5.0 | 51 | 10.2 | 222 | 0 | events | yes | occlusion_crowding |
| held-out | IC-acs_ec-clip_00001_00082-neg | indoorcrowd | 5.0 | 82 | 16.4 | 2191 | 0 | events | yes | occlusion_crowding |
| held-out | IC-acs_ec-clip_00026_00091-neg | indoorcrowd | 5.0 | 41 | 8.2 | 539 | 0 | events | yes | occlusion_crowding |
| held-out | IC-ie_central-clip_00001_00061-neg | indoorcrowd | 5.0 | 61 | 12.2 | 356 | 0 | events | yes | clear |
| held-out | IC-ie_central-clip_00001_00094-neg | indoorcrowd | 5.0 | 94 | 18.8 | 687 | 0 | events | yes | clear |
| held-out | IC-ie_central-clip_00001_00381-neg | indoorcrowd | 5.0 | 381 | 76.2 | 2039 | 0 | events | yes | clear |
| held-out | IC-r_central-clip_00001_00090-neg | indoorcrowd | 5.0 | 90 | 18.0 | 550 | 0 | events | yes | occlusion_crowding |
| held-out | IC-r_central-clip_00001_00134-neg | indoorcrowd | 5.0 | 134 | 26.8 | 1147 | 0 | events | yes | occlusion_crowding |
| held-out | IC-r_central-clip_00001_00156-neg | indoorcrowd | 5.0 | 156 | 31.2 | 872 | 0 | events | yes | occlusion_crowding |

## Known biases / limitations
MOT17 is mostly outdoor street/mall footage with different camera heights than a corridor; indoor clips are from one or few locations and will not cover all lighting, clothing or crowd types. Results do not generalise beyond the conditions listed in the manifest.
