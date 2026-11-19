from iwm import load_config
from iwm.alerts import AlertEngine
from iwm.worldmodel import PairRisk

CFG = load_config()


def risk(p, ttc):
    return [PairRisk("fl", "wk", p, ttc, 5.0)]


def test_tier_rules():
    e = AlertEngine(CFG)
    assert e.candidate(0.2, 1.0) == 0
    assert e.candidate(0.4, 4.0) == 1
    assert e.candidate(0.7, 2.5) == 2
    assert e.candidate(0.9, 1.5) == 3


def test_needs_rising_frames_before_alert():
    e = AlertEngine(CFG)
    assert e.update(0.0, risk(0.7, 2.5)) == []
    assert e.update(0.1, risk(0.7, 2.5)) == []
    alerts = e.update(0.2, risk(0.7, 2.5))
    assert len(alerts) == 1 and alerts[0].tier == "warning"


def test_single_spike_does_not_alert():
    e = AlertEngine(CFG)
    e.update(0.0, risk(0.9, 1.0))
    assert e.update(0.1, risk(0.1, 9.0)) == []
    assert e.tier_of("fl", "wk") == "none"


def test_holds_tier_then_drops():
    e = AlertEngine(CFG)
    for i in range(3):
        e.update(i * 0.1, risk(0.7, 2.5))
    e.update(0.5, risk(0.1, 9.0))
    assert e.tier_of("fl", "wk") == "warning"  # still held
    e.update(1.6, risk(0.1, 9.0))
    assert e.tier_of("fl", "wk") == "none"  # dropped after hold_s
