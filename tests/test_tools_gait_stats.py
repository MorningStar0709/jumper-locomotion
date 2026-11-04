"""`tools/gait_stats.py`: the numbers that answer "is it walking the gait we think".

The one thing these tests exist to prevent is the per-foot relative threshold
that produced a wrong duty factor for two of the six feet: on this robot the
front palms carry about twice the load of the middle and rear tips, so a
threshold derived from each foot's own peak lands above a normally loaded leg.
"""

from __future__ import annotations

import csv

import pytest

import gait_stats


def write_csv(tmp_path, frames, name="measure.csv"):
    """`frames` is a list of (t, [six forces]); the other measure columns are absent."""
    header = ["t"] + [f"force/{foot}_pad_link_meshcol" for foot in gait_stats.FEET]
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t, forces in frames:
            writer.writerow([t, *forces])
    return path


def alternating(period_frames=8, seconds=2.0, loaded=30.0, rest=0.0, rate=50.0):
    """A clean tripod walk: LF+RM+LR, then RF+LM+RR, and so on."""
    first = {"LF", "RM", "LR"}
    frames = []
    for index in range(int(seconds * rate)):
        on = first if (index // period_frames) % 2 == 0 else set(gait_stats.FEET) - first
        frames.append((index / rate, [loaded if foot in on else rest for foot in gait_stats.FEET]))
    return frames


def test_the_six_force_columns_come_back_in_feet_order(tmp_path):
    path = write_csv(tmp_path, [(0.0, [1, 2, 3, 4, 5, 6])])
    _, rows = gait_stats.read_forces(path)
    assert rows == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_a_foot_loaded_half_the_time_has_duty_one_half(tmp_path):
    frames = alternating(period_frames=10, seconds=2.0)      # tripod swaps every 10 frames
    _, rows = gait_stats.read_forces(write_csv(tmp_path, frames))
    assert gait_stats.duty(rows)["LF"] == pytest.approx(0.5, abs=0.02)


def test_a_clean_alternation_is_all_tripod_and_nothing_else(tmp_path):
    frames = alternating()
    _, rows = gait_stats.read_forces(write_csv(tmp_path, frames))
    shares = gait_stats.tripod_shares(rows)
    assert shares["LF+RM+LR"] == pytest.approx(0.5, abs=0.03)
    assert shares["RF+LM+RR"] == pytest.approx(0.5, abs=0.03)
    assert shares["other three-foot"] == 0.0
    assert shares["both tripods down"] == 0.0


def test_one_cycle_per_two_alternations_gives_the_step_frequency(tmp_path):
    frames = alternating(period_frames=8, seconds=2.0)       # 0.16 s per tripod = 3.125 Hz
    times, rows = gait_stats.read_forces(write_csv(tmp_path, frames))
    assert gait_stats.step_frequency(times, rows) == pytest.approx(3.125, abs=0.05)


def test_a_two_foot_transition_does_not_hide_the_alternation(tmp_path):
    """Real captures spend ~12% of frames in a two-foot handover.

    Those frames carry no tripod label; skipping them must not break the count,
    or a real walk reports zero hertz while both tripods are clearly down 43%
    of the time each.
    """
    first = {"LF", "RM", "LR"}
    frames = []
    for index in range(100):
        phase = (index // 10) % 2
        if index % 10 == 9:                                   # one handover frame
            on = {"LF", "RF"}
        else:
            on = first if phase == 0 else set(gait_stats.FEET) - first
        frames.append((index / 50, [30.0 if foot in on else 0.0 for foot in gait_stats.FEET]))
    times, rows = gait_stats.read_forces(write_csv(tmp_path, frames))
    assert gait_stats.step_frequency(times, rows) == pytest.approx(2.5, abs=0.15)


def test_a_light_but_loaded_foot_is_not_airborne(tmp_path):
    """The regression: 8 N on a foot whose walk carries 47 N is a foot on the ground.

    A per-foot threshold of 20% of 47 N is 9.4 N and calls it airborne; a
    threshold of 20% of the 8 N foot's own peak is 1.6 N and calls it loaded. The
    same physical state cannot have two answers -- the threshold is absolute.
    """
    frames = [(0.0, [47.0, 47.0, 8.0, 8.0, 8.0, 8.0]), (0.02, [47.0] * 6)]
    _, rows = gait_stats.read_forces(write_csv(tmp_path, frames))
    assert gait_stats.duty(rows)["LM"] == 1.0


def test_the_support_histogram_covers_every_frame(tmp_path):
    _, rows = gait_stats.read_forces(write_csv(tmp_path, alternating()))
    assert sum(gait_stats.support_shares(rows).values()) == pytest.approx(1.0)


def test_a_missing_force_column_is_named(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("t,force/LF_palm_link_meshcol\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RF"):
        gait_stats.read_forces(path)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_a_threshold_that_is_not_a_usable_number_is_refused(tmp_path, value, capsys):
    """`nan` 让每只脚都算腾空、负数让每只脚永远着地，两种都会打印出像结果的报告。"""
    path = write_csv(tmp_path, alternating())

    assert gait_stats.main(["--threshold", value, str(path)]) == 2
    assert "greater than zero" in capsys.readouterr().err


def test_a_threshold_word_that_is_not_a_number_is_still_refused(tmp_path, capsys):
    path = write_csv(tmp_path, alternating())

    assert gait_stats.main(["--threshold", "two", str(path)]) == 2
    assert "wants a number of newtons" in capsys.readouterr().err


def test_the_report_round_trips_a_file(tmp_path):
    body = gait_stats.report(write_csv(tmp_path, alternating()))
    assert body["frames"] == 100
    assert body["tripods"]["LF+RM+LR"] == pytest.approx(0.5, abs=0.03)
    assert set(body["duty"]) == set(gait_stats.FEET)


def test_a_threshold_of_zero_is_refused(tmp_path, capsys):
    """`--threshold 0` 让 `force >= 0` 永远为真：占空比 100%、单足支撑 100%。"""
    path = write_csv(tmp_path, [(0.0, [20, 20, 20, 20, 20, 20])])
    assert gait_stats.main(["--threshold", "0", str(path)]) == 2
    assert "greater than zero" in capsys.readouterr().err
