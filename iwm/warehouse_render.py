"""Draw the warehouse with Pillow (for the browser UI): overview and zoomed-in view.

Same look as demo_warehouse.py (pygame), so the UI and the desktop demo match.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .warehouse import AISLES_Y, HEIGHT, WALKS_X, WIDTH, ZONES

RACK, RACK_EDGE, AISLE, WALK, STRIPE = (176, 186, 200), (148, 160, 178), (226, 232, 240), (236, 240, 245), (250, 204, 21)
INK, QUIET, WHITE = (15, 23, 42), (100, 116, 139), (255, 255, 255)
FORK, WORK = (234, 88, 12), (37, 99, 235)
TIER = {"none": (22, 163, 74), "advisory": (202, 138, 4), "warning": (234, 88, 12), "critical": (220, 38, 38)}
ZOOM_SPAN = 14.0  # metres shown in the zoom view


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


class View:
    """Maps a floor rectangle (metres) onto an image (pixels)."""

    def __init__(self, x0, y0, span_x, span_y, w, h):
        self.x0, self.y0, self.span_y, self.w, self.h = x0, y0, span_y, w, h
        self.s = w / span_x

    def px(self, x, y):
        return (x - self.x0) * self.s, (self.y0 + self.span_y - y) * self.s

    def m(self, d):
        return max(1.0, d * self.s)


def _floor(d: ImageDraw.ImageDraw, v: View, labels: bool) -> None:
    d.rectangle([0, 0, v.w, v.h], fill=RACK)
    for y in AISLES_Y.values():
        a, b = v.px(0, y + 1.5), v.px(WIDTH, y - 1.5)
        d.rectangle([a, b], fill=AISLE)
    for x in WALKS_X.values():
        a, b = v.px(x - 1.25, HEIGHT), v.px(x + 1.25, 0)
        d.rectangle([a, b], fill=WALK)
        step = max(6, int(v.m(0.8)))
        for edge in (x - 1.25, x + 1.25):
            ex, _ = v.px(edge, 0)
            for yy in range(0, v.h, step * 2):
                d.line([(ex, yy), (ex, yy + step)], fill=STRIPE, width=max(2, int(v.m(0.08))))
    xs = [0.0] + [e for x in WALKS_X.values() for e in (x - 1.25, x + 1.25)] + [WIDTH]
    ys = [0.0] + [e for y in sorted(AISLES_Y.values()) for e in (y - 1.5, y + 1.5)] + [HEIGHT]
    for i in range(0, len(xs), 2):
        for j in range(0, len(ys), 2):
            d.rectangle([v.px(xs[i], ys[j + 1]), v.px(xs[i + 1], ys[j])], outline=RACK_EDGE, width=1)
    if labels:
        f = font(13)
        for name, y in AISLES_Y.items():
            d.text(v.px(0.4, y + 1.3), f"Aisle {name}", fill=QUIET, font=f)
        for name, x in WALKS_X.items():
            d.text(v.px(x - 1.1, HEIGHT - 0.2), f"Walk {name}", fill=QUIET, font=f)


def _paths(d, v: View, session, frame, only_pair=None) -> None:
    for r in frame.risks:
        if not r.paths or (only_pair and (r.forklift, r.worker) != only_pair):
            continue
        tier = session.engine.tier_of(r.forklift, r.worker)
        if tier == "none" and (only_pair is None or r.p_collision < 0.1):
            continue
        for path in r.paths.values():
            pts = [v.px(x, y) for x, y in path[::2]]
            if len(pts) > 1:
                d.line(pts, fill=TIER[tier], width=3, joint="curve")
                x, y = pts[-1]
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=TIER[tier])


def _agents(d, v: View, frame, detail: bool) -> None:
    f = font(15 if detail else 12)
    for tr in frame.tracks:
        x, y = v.px(tr.x, tr.y)
        if tr.type == "forklift":
            w, h = v.m(2.4), v.m(1.3)
            d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], max(2, h / 4), fill=FORK, outline=WHITE)
        else:
            r = max(5.0, v.m(0.45))
            d.ellipse([x - r, y - r, x + r, y + r], fill=WORK, outline=WHITE)
        label = f"{tr.id}  {np.hypot(tr.vx, tr.vy):.1f} m/s" if detail else tr.id
        d.text((x + 8, y - (28 if detail else 18)), label, fill=INK, font=f)


def render_overview(session, frame, focus=None, contacts=(), px_per_m: int = 25) -> Image.Image:
    """Whole warehouse with risk rings per corner; `focus` = ((x, y), zone) draws the zoom area."""
    w, h = int(WIDTH * px_per_m), int(HEIGHT * px_per_m)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    v = View(0, 0, WIDTH, HEIGHT, w, h)
    _floor(d, v, labels=True)
    f = font(13, bold=True)
    for name, (x, y) in ZONES.items():
        z = frame.zones[name]
        cx, cy = v.px(x, y)
        r = v.m(2.2)
        colour = TIER[z.tier] if z.tier != "none" else QUIET
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=4 if z.tier != "none" else 1)
        d.text((cx + v.m(1.6), cy + v.m(1.5)), name, fill=colour, font=f)
    _paths(d, v, session, frame)
    _agents(d, v, frame, detail=False)
    for c in contacts:
        x, y = v.px(*c.xy)
        d.line([(x - 10, y - 10), (x + 10, y + 10)], fill=TIER["critical"], width=4)
        d.line([(x - 10, y + 10), (x + 10, y - 10)], fill=TIER["critical"], width=4)
    if focus:
        (fx, fy), _ = focus
        d.rectangle([v.px(fx - ZOOM_SPAN / 2, fy + ZOOM_SPAN / 2), v.px(fx + ZOOM_SPAN / 2, fy - ZOOM_SPAN / 2)],
                    outline=INK, width=2)
    return img


def render_zoom(session, frame, focus, size: int = 520) -> Image.Image:
    """Zoomed-in view around `focus` with a header (zone, tier, risk, time to contact) and footer (pair, fix)."""
    (fx, fy), zone = focus
    z = frame.zones.get(zone) if zone else None
    tier = z.tier if z else "none"
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    v = View(fx - ZOOM_SPAN / 2, fy - ZOOM_SPAN / 2, ZOOM_SPAN, ZOOM_SPAN, size, size)
    _floor(d, v, labels=False)
    _paths(d, v, session, frame, only_pair=z.pair if z and z.pair else None)
    _agents(d, v, frame, detail=True)
    d.rectangle([0, 0, size - 1, size - 1], outline=TIER[tier], width=6)
    d.rectangle([0, 0, size, 40], fill=TIER[tier])
    title = f"ZOOM {zone or 'point'}"
    if z and z.tier != "none":
        if z.ttc <= 0.05:
            title += f"  ·  {z.tier.upper()}  ·  CONTACT NOW"
        elif z.p < 0.3:
            title += f"  ·  {z.tier.upper()}  ·  clearing"
        else:
            title += f"  ·  {z.tier.upper()}  {z.p:.0%}  ·  contact in {z.ttc:.1f} s"
    else:
        title += "  ·  CLEAR"
    d.text((12, 10), title, fill=WHITE, font=font(17, bold=True))
    if z and z.pair:
        reasons = session.reasons.get(z.pair, [])
        foot = f"{z.pair[0]} vs {z.pair[1]}" + (f"  ·  fix: {reasons[0][0]}" if reasons and reasons[0][1] > 0 else "")
        d.rectangle([0, size - 32, size, size], fill=INK)
        d.text((12, size - 25), foot[:70], fill=WHITE, font=font(14))
    return img
