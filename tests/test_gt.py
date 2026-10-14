from sightline.evaluation.gt import load_mot_gt


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
