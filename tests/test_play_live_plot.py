"""`play.py --measure`: when the live plot window is allowed to open.

`--headless` is the switch that says nobody is watching, and it already keeps the
3D viewer away. The measurement window ignored it, so a headless replay on a
desktop -- an agent's run, a script's run -- opened a matplotlib window on
somebody's screen and left it there. These pin the decision in one place.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="play imports torch")

from play import _wants_a_window  # noqa: E402


def _args(headless: bool, live_plot: bool) -> SimpleNamespace:
    return SimpleNamespace(headless=headless, live_plot=live_plot)


@pytest.mark.parametrize(
    ("headless", "live_plot", "expected"),
    [
        (True, False, False),  # headless: nobody is watching
        (False, False, True),  # interactive replay: that is what it is for
        (True, True, True),  # asked for explicitly, despite --headless
        (False, True, True),
    ],
)
def test_the_window_opens_only_for_a_human(
    headless: bool, live_plot: bool, expected: bool
) -> None:
    assert _wants_a_window(_args(headless, live_plot)) is expected


def test_play_hands_that_decision_to_the_monitor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decision and the monitor are each pinned; this pins the wire between.

    Both halves were green while `play.py` passed nothing: `_wants_a_window`
    answered correctly and `JointMonitor(plot=False)` obeyed, and the call site
    could still have said `plot=True`. That is how a headless replay kept opening
    a window through a fix that looked tested.
    """
    import play
    from mjrl.viewer import monitor as monitor_module

    seen: dict[str, object] = {}

    def fake_make_monitor(entity, index, env, out, *, plot=True):
        seen.update(entity=entity, index=index, out=out, plot=plot)
        return None

    monkeypatch.setattr(monitor_module, "make_monitor", fake_make_monitor)
    monkeypatch.setattr(play, "_watched", lambda env: ("the-entity", 0))

    play._monitor_for(_args(headless=True, live_plot=False), object(), tmp_path)
    assert seen["plot"] is False

    play._monitor_for(_args(headless=False, live_plot=False), object(), tmp_path)
    assert seen["plot"] is True
