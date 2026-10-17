from sightline.evaluation.gt import InScope, load_mot_gt


def test_out_of_scope_people_become_ignore_regions(tmp_path):
    # frame height 1000 -> in scope needs height >= 100 px and visibility >= 0.65
    rows = [
        "1,1,0,0,40,150,1,1,1.0",    # tall, visible: in scope
        "1,2,100,0,20,60,1,1,1.0",   # too short: out of scope
        "1,3,200,0,40,150,1,1,0.4",  # too occluded: out of scope
        "1,4,300,0,40,100,1,1,0.65", # exactly on both limits: in scope
    ]
    p = tmp_path / "gt.txt"
    p.write_text("\n".join(rows) + "\n")
    gt, ignore = load_mot_gt(p, in_scope=InScope(0.10, 0.65, 1000))
    assert sorted(tid for tid, _b in gt[1]) == [1, 4]
    assert len(ignore[1]) == 2
    gt_all, ignore_all = load_mot_gt(p)
    assert len(gt_all[1]) == 4 and not ignore_all


def test_mot_gt_scores_pedestrians_ignores_distractors_and_drops_non_people(tmp_path):
    rows = [
        "1,1,0,0,10,20,1,1,1.0",    # pedestrian, conf 1: scored
        "1,2,0,0,10,20,0,1,1.0",    # pedestrian, conf 0: ignore region
        "1,3,50,0,10,20,0,7,1.0",   # static person (distractor): ignore region
        "1,4,90,0,10,20,0,12,1.0",  # reflection (distractor): ignore region
        "1,5,200,0,40,20,0,3,1.0",  # car: dropped, NOT an ignore region
        "1,6,300,0,40,20,0,10,1.0", # occluder on the ground: dropped
    ]
    p = tmp_path / "gt.txt"
    p.write_text("\n".join(rows) + "\n")
    gt, ignore = load_mot_gt(p)
    assert [tid for tid, _b in gt[1]] == [1]
    assert sorted(ignore[1]) == sorted([(0.0, 0.0, 10.0, 20.0), (50.0, 0.0, 60.0, 20.0), (90.0, 0.0, 100.0, 20.0)])


def test_unknown_visibility_is_kept_and_judged_in_scope_by_height_alone(tmp_path):
    p = tmp_path / "gt.txt"
    # frame 1: tall person, visibility unknown (-1); short person, unknown; tall person, known 0.3
    p.write_text("1,1,0,0,10,50,1,1,-1\n1,2,0,0,10,5,1,1,-1\n1,3,0,0,10,50,1,1,0.3\n")
    gt, ignore = load_mot_gt(p)
    assert [t for t, _b in gt[1]] == [1, 2, 3]  # unknown visibility is not dropped
    gt_in, ignore_in = load_mot_gt(p, in_scope=InScope(0.10, 0.65, 100))
    assert [t for t, _b in gt_in[1]] == [1]  # tall + unknown in scope; short and known-occluded ignored
    assert len(ignore_in[1]) == 2
