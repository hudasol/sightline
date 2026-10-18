"""The headline numbers in the hand-written docs must match the saved result files.

docs/RESULTS.md is generated, but README, EVALUATION, MODEL_CARD and RETROSPECTIVE quote numbers by hand.
This test recomputes each quoted number from results/ and checks the docs still say it, so a doc cannot
drift from the results without failing the build. It never changes a number.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
pytestmark = pytest.mark.skipif(not (R / "final_metrics.json").is_file(), reason="no held-out results saved")


def _j(rel):
    return json.loads((R / rel).read_text())


def _doc(name):
    """The doc's text with bold markers and "of the" normalised, so "**10 of the 17**" reads as "10 of 17"."""
    text = (ROOT / name).read_text(encoding="utf-8").replace("**", "")
    return " ".join(text.replace(" of the ", " of ").split())


def _expected() -> dict[str, list[str]]:
    f = _j("final_metrics.json")
    det = f["detection_in_scope_by_dataset"]["indoorcrowd"]
    trk, base = f["tracking"]["bytetrack"]["ALL"], f["tracking"]["iou"]["ALL"]
    ev, safety = f["events"]["bytetrack"]["ALL"], f["safety"]["ALL"]
    runs = _j("runtime/heldout_offline/summary.json")["frozen_k3"]["throughput_fps"]
    run1 = _j("runtime_report.json")["summary"]["throughput_fps"]
    lounge = ("IC-acs_ec-clip_00001_00082", "IC-acs_ec-clip_00026_00091")
    missed = sum(len(_j(f"per_clip/bytetrack/{c}.json")["missed_events"]) for c in lounge)
    fc = _j("safety/false_clear_by_clip.json")["clips"]
    lounge_fc = sum(fc[c]["false_clear_frames"] for c in lounge)
    return {
        "detection P/R (IndoorCrowd, in scope)": [f"{det['precision']:.3f}", f"{det['recall']:.3f}"],
        "IDF1 ByteTrack": [f"{trk['idf1']:.3f}"],
        "IDF1 baseline": [f"{base['idf1']:.3f}"],
        "ID switches": [f"{trk['num_switches']} vs {base['num_switches']}"],
        "event P/R": [f"{ev['event_precision']:.3f}", f"{ev['event_recall']:.3f}"],
        "false CLEAR": [f"{safety['false_clear_frames']} of {safety['occupied_frames']}"],
        "missed events": [f"{ev['events_fn']} missed"],
        "ACS-EC lounge missed": [f"{missed} of {ev['events_fn']}"],
        "ACS-EC lounge false CLEAR": [f"{lounge_fc} of {safety['false_clear_frames']}"],
        "throughput run 1 and median": [f"{run1:.2f}", f"{runs['median']:.2f}"],
    }


@pytest.mark.parametrize("doc", ["README.md", "docs/EVALUATION.md", "docs/RETROSPECTIVE.md"])
def test_quoted_headline_numbers_match_the_saved_results(doc):
    text = _doc(doc)
    exp = _expected()
    # each doc quotes a subset; every number a doc quotes for a claim below must be the saved value
    checks = {
        "README.md": ["IDF1 ByteTrack", "event P/R", "false CLEAR", "ACS-EC lounge missed", "ACS-EC lounge false CLEAR"],
        "docs/EVALUATION.md": [k for k in exp if k != "ACS-EC lounge missed"],  # it says "10 are on ..." under "17 misses"
        "docs/RETROSPECTIVE.md": ["IDF1 ByteTrack", "event P/R", "ACS-EC lounge missed", "ACS-EC lounge false CLEAR"],
    }[doc]
    missing = [(k, v) for k in checks for v in exp[k] if v not in text]
    assert not missing, f"{doc} does not quote the saved values: {missing}"


def test_model_card_quotes_the_acs_ec_share_and_false_clear():
    text, exp = _doc("docs/MODEL_CARD.md"), _expected()
    for v in exp["ACS-EC lounge missed"] + exp["ACS-EC lounge false CLEAR"] + exp["false CLEAR"]:
        assert v in text
