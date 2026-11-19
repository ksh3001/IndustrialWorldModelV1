from iwm import load_config
from iwm.demo import DemoSession
from iwm.plots import floor_plot, risk_timeline
from iwm.report import template_brief
from iwm.sim import WhatIf

CFG = load_config()


def test_blind_corner_warns_early_and_alerts_prevent_contact():
    s = DemoSession(CFG, "blind_corner", 1)
    s.run()
    sm = s.summary()
    assert sm["contact_s"] is not None
    assert sm["lead_first_alert_s"] >= 2.5 > sm["lead_detection_only_s"]

    s2 = DemoSession(CFG, "blind_corner", 1, WhatIf(workers_react_to_alerts=True))
    s2.run()
    assert s2.summary()["contact_s"] is None


def test_plots_and_brief_render():
    s = DemoSession(CFG, "blind_corner", 1)
    s.run()
    floor_plot(s, 10)
    risk_timeline(s)
    brief = template_brief(s.summary(), s.alerts[0].to_dict(), s.last_reasons)
    assert "Suggested fix" in brief
