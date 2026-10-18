# Demo footage: candidate clips and their licence

**Status: candidates only. Nothing has been downloaded or used. A clip is used only after Huda approves it.**

The demo video must not show MOT17 or IndoorCrowd frames (DATA_CARD: no frames published; IndoorCrowd's licence
prohibits redistribution and surveillance use). It needs footage that may be shown on screen, has no identifiable
faces, and has nothing to do with police or security. Checked on 2026-10-18.

## The licence: Pexels License
Source: https://www.pexels.com/license/ (read 2026-10-18; the page shows no "last updated" date). In summary, in
our words (see the page for the binding text):
- **Allowed:** free use of photos and videos, including commercial use. Editing and modification are allowed.
  Attribution is not required (it is "appreciated").
- **Not allowed:**
  - showing identifiable people in a bad light or in an offensive way;
  - selling unaltered copies;
  - implying that a person or brand endorses you;
  - redistributing or selling the media on other stock or wallpaper platforms;
  - using the media as a trademark or business name.
- **Not addressed:** model or property releases. That is one more reason to pick a clip with no identifiable faces.

How a demo would use it: a modified derivative (detection boxes, track ids, corridor overlay), shown in a project
demo video, crediting the creator. That fits the terms above. It does not put any person in a negative light: the
overlay draws anonymous boxes only.

## Candidate 1 (recommended): "Blur People Walking"
- **Page:** https://www.pexels.com/video/blur-people-walking-6233170/
- **Creator:** Neugiernase (https://www.pexels.com/@neugiernase-12652911/).
- **Licence label on the page:** "Free" (Pexels License).
- **Clip:** 0:36, 3840x2160, 25 fps, uploaded 2020-12-19. Pexels describes it as people walking in an indoor
  subway corridor.
- **What was checked:** the page text (no mention of faces, police, security or surveillance), and the page's
  preview frame, viewed in a browser and not downloaded. It shows:
  - a camera close to the floor, looking down a corridor;
  - several people walking, all strongly out of focus;
  - no identifiable faces.

  That is a robot's-eye view, which suits the scenario.
- **Not checked:** the full 36 s. Faces in other frames could be clearer. Review the whole clip before approving it.
- **Expected limit for the demo:** the people are out of focus, so the detector may be less confident. That is fine
  for showing confidence, health and events, but it is not an accuracy claim: the clip has no ground truth.

## Other candidates (page text checked only; no frame seen)
| Clip | Page | Creator | Format | Note |
|---|---|---|---|---|
| Back view of a woman walking down a hallway | https://www.pexels.com/video/back-view-of-a-woman-walking-down-a-hallway-16252758/ | Unam Mayanja | 0:16, 1080x1920 (portrait), 29.97 fps | Back view, one person |
| Silhouettes in modern architecture | https://www.pexels.com/video/silhouettes-in-modern-architecture-40099259/ | Chinar Minar | 0:06, portrait, 29.97 fps | Silhouettes; very short |
| Back view of a man and a woman walking while holding hands in the hallway | https://www.pexels.com/video/back-view-of-a-man-and-a-woman-walking-while-holding-hands-in-the-hallway-6998323/ | Pavel Danilyuk | 0:24, 1080x1920, 25 fps | Hospital hallway: a health setting, so less suitable |

Rejected without further checks: airport and terminal clips (a security context).

## The clip Huda found
Huda mentioned a Pexels hallway clip she had found; its URL is not in the repository. Send the URL and it gets
the same check before any decision.

## After approval (not done yet)
1. Download the approved clip by hand from its page to `data/demo/` (git ignored). Record its file name, size and
   SHA-256 here.
2. Draw its corridor once with `sightline draw-corridor` and save it as `configs/corridors/demo.yaml`.
3. Credit the creator in the demo video's description and in `docs/DEMO.md`.
