from pathlib import Path

import pytest

from sightline.report_html import build_html

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def test_report_page_needs_the_held_out_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="final_metrics.json"):
        build_html(tmp_path)


@pytest.mark.skipif(not (RESULTS / "final_metrics.json").is_file(), reason="no held-out results saved")
def test_report_page_is_self_contained_and_shows_pass_and_fail():
    page = build_html(RESULTS)
    assert page.startswith("<!doctype html>")
    assert "<script" not in page and "http://" not in page and "https://" not in page  # no code, no network
    assert "data:image/png;base64," in page  # the stress plot is embedded, not linked
    assert 'class="pass">PASS' in page and 'class="fail">FAIL' in page
    assert "Tracker ablation" in page and "One corridor event" in page
