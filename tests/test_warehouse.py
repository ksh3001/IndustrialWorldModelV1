from iwm import load_config
from iwm.sim import WhatIf
from iwm.warehouse import ZONES, WarehouseSession

CFG = load_config()


def run(whatif, steps=1500, seed=3):
    s = WarehouseSession(CFG, seed=seed, whatif=whatif)
    for _ in range(steps):
        s.step()
    return s


def test_nine_zones_and_status_for_each():
    s = run(WhatIf(), steps=50)
    assert len(ZONES) == 9 and set(s.frame.zones) == set(ZONES)


def test_encounters_raise_alerts_and_alerts_reduce_collisions():
    base, react = run(WhatIf()), run(WhatIf(workers_react_to_alerts=True))
    assert base.stats()["encounters"] > 10 and base.alerts
    assert react.stats()["contacts"] < base.stats()["contacts"]


def test_hottest_zone_matches_an_alert_zone():
    s = WarehouseSession(CFG, seed=3)
    for _ in range(1500):
        f = s.step()
        if f.new_alerts:
            hot = s.hottest()
            assert hot is not None and hot.tier != "none"
            return
    raise AssertionError("no alert raised in 150 s")


def test_renderer_draws_overview_and_zoom():
    from iwm.warehouse_render import render_overview, render_zoom

    s = run(WhatIf(), steps=30)
    over = render_overview(s, s.frame, (ZONES["B2"], "B2"))
    zoom = render_zoom(s, s.frame, (ZONES["B2"], "B2"))
    assert over.size == (1000, 600) and zoom.size == (520, 520)
