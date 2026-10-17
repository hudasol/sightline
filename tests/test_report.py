from sightline.report import build


def test_report_with_no_results_says_so_and_invents_nothing(tmp_path):
    text = build(tmp_path)
    assert text.startswith("# Results (generated)")
    assert "## The bar" not in text and "PASS" not in text  # no final metrics, no claims


def test_report_names_its_source_file_in_every_section(tmp_path):
    import json

    (tmp_path / "ablation").mkdir()
    (tmp_path / "ablation" / "ablation_heldout.json").write_text(json.dumps({"variants": {"x": {
        "idf1": 0.7, "id_switches": 3, "idf1_in_scope": 0.8, "event_precision": 0.9, "event_recall": 0.95,
        "latency_frames_median": 1.0, "events_unresolved": 2, "negative_clip_false_events": 0}}}))
    text = build(tmp_path)
    assert "`results/ablation/ablation_heldout.json`" in text and "| x | 0.700 | 3 | 0.800 |" in text


def test_latency_plot_reads_the_key_evaluate_writes(tmp_path):
    import json

    from sightline.plots import plot_latency

    (tmp_path / "bytetrack").mkdir()
    (tmp_path / "bytetrack" / "c.json").write_text(json.dumps({"event_latencies_frames": [0, 1, 1, 3]}))
    assert plot_latency(tmp_path, tmp_path / "lat.png") == tmp_path / "lat.png" and (tmp_path / "lat.png").is_file()
