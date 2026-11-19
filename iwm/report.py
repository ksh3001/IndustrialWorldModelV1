"""Incident brief: turn an alert and its numbers into plain English.

Uses Azure OpenAI when AZURE_OPENAI_ENDPOINT / _API_KEY / _DEPLOYMENT are set;
otherwise falls back to a template so the demo never breaks.

    python -m iwm.report          # brief for the blind-corner demo near-miss
"""

from __future__ import annotations

import os

PROMPT = """You are a site safety assistant. Write a short incident brief for a near-miss
detected by an AI world model watching CCTV. Use exactly these headings, one line each:
What happened / When and where / Risk / Most likely cause / Suggested fix.
Be factual, use only the data given, no blame on individuals. Data:
{facts}"""


def _facts(summary: dict, alert: dict | None, reasons: list | None) -> str:
    lines = [f"- {k}: {v}" for k, v in summary.items()]
    if alert:
        lines.append(f"- first alert: tier={alert['tier']}, p_collision={alert['p_collision']}, "
                     f"predicted time to contact={alert['ttc_s']} s, at t={alert['t']} s")
    if reasons:
        lines.append("- counterfactual risk reductions: " + "; ".join(f"{n}: -{d:.0%}" for n, d in reasons))
        lines.append(f"- suggested site fix: {site_fix(reasons)}")
    return "\n".join(lines)


# Turn a moment-in-time counterfactual (worldmodel.explain) into a site fix.
SITE_FIX = {
    "forklift slows": "forklift speed limit of 1 m/s within 5 m of the corner",
    "worker stops": "stop-and-look line and convex mirror at the walkway entrance",
    "forklift stops": "forklift stop line before the crossing (pedestrian priority)",
}


def site_fix(reasons: list | None) -> str:
    if not reasons or reasons[0][1] <= 0:
        return SITE_FIX["forklift slows"]
    name = reasons[0][0]
    return next((fix for key, fix in SITE_FIX.items() if name.startswith(key)), name)


def template_brief(summary: dict, alert: dict | None = None, reasons: list | None = None) -> str:
    contact = summary.get("contact_s")
    lead = summary.get("lead_first_alert_s")
    fix = site_fix(reasons)
    happened = ("Worker and forklift came within contact distance" if contact is not None
                else "Worker and forklift paths were predicted to cross; no contact occurred")
    return "\n".join([
        f"What happened: {happened} at the blind corner (closest {summary.get('closest_m')} m).",
        f"When and where: t = {contact if contact is not None else alert['t'] if alert else '-'} s, "
        f"aisle/walkway intersection.",
        f"Risk: first alert {lead if lead is not None else '-'} s before contact"
        + (f" (tier {alert['tier']}, p = {alert['p_collision']})" if alert else "") + ".",
        "Most likely cause: limited visibility at the corner combined with forklift speed.",
        f"Suggested fix: {fix}.",
    ])


def incident_brief(summary: dict, alert: dict | None = None, reasons: list | None = None) -> str:
    endpoint, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not (endpoint and key and deployment):
        return template_brief(summary, alert, reasons)
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                             api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        resp = client.chat.completions.create(
            model=deployment, temperature=0.2, max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(facts=_facts(summary, alert, reasons))}])
        return resp.choices[0].message.content.strip()
    except Exception as e:  # network or key problems: never break the demo
        return template_brief(summary, alert, reasons) + f"\n(template used: {e.__class__.__name__})"


if __name__ == "__main__":
    from . import load_config
    from .demo import DemoSession

    s = DemoSession(load_config(), "blind_corner", 1)
    s.run()
    first = s.alerts[0].to_dict() if s.alerts else None
    print(incident_brief(s.summary(), first, s.last_reasons))


SHIFT_PROMPT = """You are a site safety assistant writing for a site manager. Summarise this shift's
near-miss data from an AI world model watching CCTV in 4 short bullet points:
overall picture, highest-risk pattern, recommended fix, what to watch next shift.
Use only the data given; no blame on individuals. Data:
{facts}"""


def executive_summary(summary: dict) -> str:
    """Shift-level summary via Azure OpenAI, with a template fallback."""
    template = "\n".join([
        f"- {summary['incidents']} of {summary['encounters']} forklift-worker encounters raised an alert; "
        f"{summary['contacts']} ended in contact (no reaction to alerts in this run).",
        f"- {summary['critical']} critical alerts: crossings at the blind corner are the dominant risk.",
        f"- Recommended fix: {summary['top_fix']}.",
        f"- Average warning before contact was {summary['avg_warning_s']} s: enable worker and driver alerts next shift.",
    ])
    endpoint, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not (endpoint and key and deployment):
        return template
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                             api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        facts = "\n".join(f"- {k}: {v}" for k, v in summary.items())
        resp = client.chat.completions.create(
            model=deployment, temperature=0.2, max_tokens=300,
            messages=[{"role": "user", "content": SHIFT_PROMPT.format(facts=facts)}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return template + f"\n(template used: {e.__class__.__name__})"
