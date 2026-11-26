from collections import defaultdict

import pytest

from berthwise.berth_assignment.commitment import (
    calibrate, sign_out, locked_blocks, commitment_digest, backtest_slips,
)
from berthwise.berth_assignment.contracts import AssignmentRequest, Booking
from berthwise.berth_assignment.planner import book
from berthwise.pipeline import run
from berthwise.tide_slices.contracts import Opening

CUTOFF = 500


def closure(call, line, closed, duration):
    return {"call": call, "line": line, "arrived": closed - duration - 10, "closed": closed,
            "boxes": 300, "reefer": 0.1, "draft": 9.0, "cranes": 2, "yard": 0.5,
            "duration": duration, "late": False}


def closures():
    rows = []
    for i in range(12):
        rows.append(closure(f"K-{i+1:02d}", "快航", 100 + 10 * i, 60))
    for i in range(8):
        rows.append(closure(f"M-{i+1:02d}", "慢航", 300 + 10 * i, 60))
    for i in range(4):
        rows.append(closure(f"M-{i+9:02d}", "慢航", 400 + 10 * i, 200))
    for i in range(2):
        rows.append(closure(f"X-{i+1:02d}", "新航", 460 + 10 * i, 60))
    return rows


def sample_calibration():
    return calibrate(closures(), CUTOFF, min_support=4, min_history=4)


def job(name="T9", **extra):
    base = dict(call=name, eta=60, expected=48, cold_boxes=10, urgent=False, manual=False,
                windows=(Opening("S2", 60, 600, .8, 3),), cranes_needed=2, max_wait=600)
    base.update(extra)
    return AssignmentRequest(**base)


def booking(call="A", **extra):
    base = dict(call=call, berth="D-1", moor=100, sail=160, wait=10,
                state="安排靠泊", reason="", cranes=2, margin=.5)
    base.update(extra)
    return Booking(**base)


def issued(call="P-1", **extra):
    base = {"call": call, "line": "航线甲", "berth": "D-1",
            "moored": 100, "promised": 200, "issued": 90}
    base.update(extra)
    return base


def test_route_with_coverage_tightens_but_never_loosens():
    cal = sample_calibration()
    assert cal.port_buffer == 140
    fast = cal.buffer_for("快航")
    assert fast.buffer == 0 and fast.tightened and fast.source == "航线回测"
    slow = cal.buffer_for("慢航")
    assert slow.buffer == cal.port_buffer and not slow.tightened


def test_small_sample_route_falls_back_to_port_caliber():
    cal = sample_calibration()
    tiny = cal.buffer_for("新航")
    assert tiny.support == 2 and tiny.route_buffer is None
    assert tiny.buffer == cal.port_buffer and tiny.source == "全港口径"
    unknown = cal.buffer_for("未知航线")
    assert unknown.buffer == cal.port_buffer and unknown.source == "全港口径"


def test_closures_after_shift_start_never_flow_back():
    base = sample_calibration()
    future = [closure("F-1", "快航", CUTOFF, 300), closure("F-2", "慢航", CUTOFF + 120, 300)]
    assert calibrate(closures() + future, CUTOFF, min_support=4, min_history=4) == base
    assert base.points == 22


def test_backtest_uses_only_earlier_closures():
    slips = backtest_slips(closures(), CUTOFF, k=7, min_history=4)
    assert set(slips) == {"快航", "慢航", "新航"}
    assert max(slips["快航"]) == 0 and max(slips["慢航"]) == 140
    with pytest.raises(ValueError, match="重复"):
        backtest_slips(closures() + [closures()[0]], CUTOFF)


def test_empty_history_uses_default_port_buffer():
    cal = calibrate([], CUTOFF)
    assert cal.port_buffer == 60 and cal.points == 0 and cal.routes == ()
    assert cal.buffer_for("任何航线").buffer == 60
    with pytest.raises(ValueError, match="下限"):
        calibrate([], CUTOFF, min_support=0)


def test_calibration_is_deterministic():
    assert sample_calibration() == sample_calibration()


def test_stretch_holds_berth_until_promised_departure():
    first, second = job("A"), job("B", eta=130)
    plain = book([first, second], 3)
    assert plain[0].sail == 120 and plain[1].moor == 135
    stretched = book([first, second], 3, stretch={"A": 90})
    assert stretched[0].sail == 120
    assert stretched[1].moor == 210


def test_stretch_minutes_are_validated():
    with pytest.raises(ValueError, match="宽放"):
        book([job()], 3, stretch={"T9": -1})
    with pytest.raises(ValueError, match="宽放"):
        book([job()], 3, stretch={"T9": 4001})


def test_issued_commitment_kept_until_original_departure():
    locked = locked_blocks([issued()], 150)
    assert locked == {"D-1": ((100, 200),)}
    opening = Opening("D-1", 60, 600, .8, 3)
    result = book([job("A", eta=150, windows=(opening,))], 3, locked)
    assert result[0].moor == 210


def test_departed_commitment_is_not_reserved():
    assert locked_blocks([issued()], 200) == {}
    assert locked_blocks([issued(promised=150)], 150) == {}


def test_overlapping_commitments_merge_into_one_block():
    locked = locked_blocks([issued("P-1"), issued("P-2", moored=150, promised=260)], 100)
    assert locked == {"D-1": ((100, 260),)}


def test_issued_commitment_validation():
    with pytest.raises(ValueError, match="晚于开班"):
        locked_blocks([issued(issued=160)], 150)
    with pytest.raises(ValueError, match="重复"):
        locked_blocks([issued(), issued()], 150)
    with pytest.raises(ValueError, match="晚于靠泊"):
        locked_blocks([issued(moored=200, promised=100)], 150)
    with pytest.raises(ValueError):
        locked_blocks([issued(berth=" ")], 150)


def test_sign_out_covers_only_booked_and_matches_buffer():
    cal = sample_calibration()
    bookings = (booking("A"), booking("B", berth=None, moor=None, sail=None,
                                      wait=None, state="继续候泊", margin=None))
    commits = sign_out(bookings, {"A": "快航", "B": "慢航"}, cal)
    assert len(commits) == 1
    assert commits[0].call == "A" and commits[0].promised == 160
    assert commits[0].buffer == cal.buffer_for("快航").buffer


def test_sign_out_requires_route_and_is_sorted():
    cal = sample_calibration()
    with pytest.raises(ValueError, match="航线"):
        sign_out((booking("A"),), {}, cal)
    fleet = (booking("B", berth="D-2", moor=120), booking("A", berth="D-1", moor=200),
             booking("C", berth="D-1", moor=100))
    lines = {"A": "慢航", "B": "慢航", "C": "慢航"}
    assert [c.call for c in sign_out(fleet, lines, cal)] == ["C", "A", "B"]


def test_digest_tracks_commitments_and_locked_blocks():
    cal = sample_calibration()
    commits = sign_out((booking("A"),), {"A": "快航"}, cal)
    digest = commitment_digest(commits, {"D-1": ((100, 200),)}, cal)
    assert digest == commitment_digest(commits, {"D-1": ((100, 200),)}, cal)
    assert digest != commitment_digest(commits, {}, cal)
    assert digest != commitment_digest((), {"D-1": ((100, 200),)}, cal)


def stage_records(result):
    return result["stages"][4]["records"]


def test_pipeline_signs_out_shift_commitments():
    result = run()
    records = stage_records(result)
    commits = [row for row in records if "promised" in row and "buffer" in row]
    booked = {row["call"]: row for row in result["rows"] if row["berth"]}
    assert len(commits) == len(booked) == result["metrics"]["已签出承诺"]
    for item in commits:
        row = booked[item["call"]]
        assert item["promised"] == row["sail"] + item["buffer"] == row["promised"]
    for row in result["rows"]:
        assert (row["promised"] is not None) == bool(row["berth"])


def test_pipeline_keeps_issued_commitments_visible_and_binding():
    result = run()
    records = stage_records(result)
    kept = {row["call"]: row for row in records if "kept" in row}
    assert set(kept) == {"PAC-148", "PAC-149"} and all(kept.values())
    rows = {row["call"]: row for row in result["rows"]}
    assert rows["PAC-152"]["moor"] == 1155
    assert rows["PAC-158"]["moor"] == 1125


def test_pipeline_caliber_evidence_shows_tightening_and_fallback():
    records = stage_records(run())
    calib = {row["line"]: row for row in records if "route_buffer" in row}
    assert calib["台湾海峡"]["tightened"]
    assert calib["台湾海峡"]["buffer"] < calib["台湾海峡"]["port_buffer"]
    assert calib["珠江航道"]["buffer"] == calib["珠江航道"]["port_buffer"]
    assert all(row["support"] >= 6 for row in calib.values())


def test_promised_blocks_never_overlap_on_a_berth():
    by_berth = defaultdict(list)
    for row in run()["rows"]:
        if row["berth"]:
            by_berth[row["berth"]].append(row)
    for berth, items in by_berth.items():
        items.sort(key=lambda row: row["moor"])
        for earlier, later in zip(items, items[1:]):
            assert later["moor"] >= earlier["promised"], berth


def test_same_shift_rerun_keeps_commitments_and_schedule():
    first, second = run(), run()
    commits_a = [row for row in stage_records(first) if "promised" in row and "buffer" in row]
    commits_b = [row for row in stage_records(second) if "promised" in row and "buffer" in row]
    assert commits_a == commits_b
    schedule = lambda result: [(row["call"], row["berth"], row["moor"], row["sail"])
                               for row in result["rows"]]
    assert schedule(first) == schedule(second)
    digest = lambda result: [row["commitment_digest"] for row in stage_records(result)
                             if "commitment_digest" in row]
    assert digest(first) == digest(second) and len(digest(first)) == 1
