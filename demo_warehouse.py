"""Warehouse overview demo in pygame: the whole site, nine blind corners, auto zoom on danger.

    python demo_warehouse.py                          # overview; alerts go to the phone if NTFY_TOPIC is set
    python demo_warehouse.py --phone wk_1             # only send worker wk_1's alerts (the phone = wk_1)
    python demo_warehouse.py --record data/warehouse_demo.mp4

What you see
  Left    the whole warehouse (zoomed out): racking, forklift aisles A-C, walkways 1-3, nine
          intersections A1..C3 ringed by their current risk, forklifts (orange) and workers (blue).
  Pop-up  when a collision is predicted, a zoomed-in view opens over the overview: the two people
          involved, their predicted paths, risk, time to contact and the top fix. It closes a few
          seconds after the risk clears.
  Right   zone grid, live counters, alert log, what-if switches.

Mouse / keys
  Click an intersection (or any point)  pin the zoom there      Click the pop-up's [x] or Esc  back to auto
  1-9  pin zone A1..C3                  0  auto zoom            SPACE  pause      + / -  speed
  A  alerts reach worker & driver       M  mirrors              S  speed limit    R  new shift   Q  quit
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pygame

from iwm import load_config
from iwm.schema import TIERS
from iwm.sim import WhatIf
from iwm.warehouse import AISLES_Y, HEIGHT, WALKS_X, WIDTH, ZONES, WarehouseSession

PX = 25  # overview pixels per metre -> 1000 x 600
OW, OH = int(WIDTH * PX), int(HEIGHT * PX)
PANEL, BAR = 380, 44
POP = 440  # pop-up size (px)
POP_SPAN = 14.0  # metres shown in the pop-up
HOLD_S = 3.0  # keep the pop-up this long after risk clears

BG, RACK, RACK_EDGE = (241, 245, 249), (176, 186, 200), (148, 160, 178)
AISLE, WALK, STRIPE = (226, 232, 240), (236, 240, 245), (250, 204, 21)
INK, QUIET, WHITE = (15, 23, 42), (100, 116, 139), (255, 255, 255)
FORK, WORK = (234, 88, 12), (37, 99, 235)
TIER = {"none": (22, 163, 74), "advisory": (202, 138, 4), "warning": (234, 88, 12), "critical": (220, 38, 38)}
ZONE_KEYS = list(ZONES)  # A1..C3 -> keys 1..9


class View:
    """Maps a rectangle of the floor (metres) onto a surface (pixels)."""

    def __init__(self, x0: float, y0: float, span_x: float, span_y: float, w: int, h: int):
        self.x0, self.y0, self.sx, self.sy, self.w, self.h = x0, y0, w / span_x, h / span_y, w, h
        self.span_y = span_y

    def px(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) * self.sx), int((self.y0 + self.span_y - y) * self.sy)

    def m(self, d: float) -> int:
        return max(1, int(d * self.sx))

    def to_floor(self, u: int, v: int) -> tuple[float, float]:
        return self.x0 + u / self.sx, self.y0 + self.span_y - v / self.sy


def draw_floor(surf, view: View, fonts, labels: bool) -> None:
    surf.fill(RACK)
    for y in AISLES_Y.values():  # forklift aisles, 3 m wide
        x0, y0 = view.px(0, y + 1.5)
        surf.fill(AISLE, pygame.Rect(x0, y0, view.m(WIDTH), view.m(3.0)))
    for x in WALKS_X.values():  # walkways, 2.5 m wide, yellow edge stripes
        x0, y0 = view.px(x - 1.25, HEIGHT)
        surf.fill(WALK, pygame.Rect(x0, y0, view.m(2.5), view.m(HEIGHT)))
        for edge in (x - 1.25, x + 1.25):
            ex, _ = view.px(edge, 0)
            step = max(6, view.m(0.8))
            for yy in range(0, view.h, step * 2):
                pygame.draw.line(surf, STRIPE, (ex, yy), (ex, yy + step), max(2, view.m(0.08)))
    # rack outlines (blocks between aisles and walkways)
    xs = [0.0] + [v for x in WALKS_X.values() for v in (x - 1.25, x + 1.25)] + [WIDTH]
    ys = [0.0] + [v for y in sorted(AISLES_Y.values()) for v in (y - 1.5, y + 1.5)] + [HEIGHT]
    for i in range(0, len(xs), 2):
        for j in range(0, len(ys), 2):
            a, b = view.px(xs[i], ys[j + 1]), view.px(xs[i + 1], ys[j])
            pygame.draw.rect(surf, RACK_EDGE, pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]), 1)
    if labels:
        for name, y in AISLES_Y.items():
            surf.blit(fonts["small"].render(f"Aisle {name}", True, QUIET), view.px(0.4, y + 1.3))
        for name, x in WALKS_X.items():
            surf.blit(fonts["small"].render(f"Walk {name}", True, QUIET), view.px(x - 1.1, HEIGHT - 0.2))


def draw_agents(surf, view: View, frame, fonts, detail: bool) -> None:
    for tr in frame.tracks:
        x, y = view.px(tr.x, tr.y)
        speed = float(np.hypot(tr.vx, tr.vy))
        if tr.type == "forklift":
            w, h = view.m(2.4), view.m(1.3)
            pygame.draw.rect(surf, FORK, pygame.Rect(x - w // 2, y - h // 2, w, h), border_radius=max(2, h // 4))
            pygame.draw.rect(surf, WHITE, pygame.Rect(x - w // 2, y - h // 2, w, h), 1, border_radius=max(2, h // 4))
        else:
            r = max(5, view.m(0.45))
            pygame.draw.circle(surf, WORK, (x, y), r)
            pygame.draw.circle(surf, WHITE, (x, y), r, 1)
        label = f"{tr.id}  {speed:.1f} m/s" if detail else tr.id
        font = fonts["med"] if detail else fonts["tiny"]
        surf.blit(font.render(label, True, INK), (x + 8, y - (26 if detail else 18)))


def draw_paths(surf, view: View, session, frame, only_pair=None) -> None:
    for r in frame.risks:
        if not r.paths:
            continue
        if only_pair and (r.forklift, r.worker) != only_pair:
            continue
        tier = session.engine.tier_of(r.forklift, r.worker)
        if tier == "none" and (only_pair is None or r.p_collision < 0.1):
            continue
        for path in r.paths.values():
            pts = [view.px(x, y) for x, y in path[::2]]
            if len(pts) > 1:
                pygame.draw.lines(surf, TIER[tier], False, pts, 3)
                pygame.draw.circle(surf, TIER[tier], pts[-1], 4)


def draw_overview(surf, view, session, frame, fonts, focus, recent_contacts) -> None:
    draw_floor(surf, view, fonts, labels=True)
    for name, (x, y) in ZONES.items():  # intersection rings coloured by risk
        z = frame.zones[name]
        cx, cy = view.px(x, y)
        colour = TIER[z.tier] if z.tier != "none" else QUIET
        pygame.draw.circle(surf, colour, (cx, cy), view.m(2.2), 4 if z.tier != "none" else 1)
        surf.blit(fonts["small"].render(name, True, colour if z.tier != "none" else QUIET), (cx + view.m(1.6), cy + view.m(1.6)))
    draw_paths(surf, view, session, frame)
    draw_agents(surf, view, frame, fonts, detail=False)
    for c in recent_contacts:
        x, y = view.px(*c.xy)
        pygame.draw.line(surf, TIER["critical"], (x - 10, y - 10), (x + 10, y + 10), 4)
        pygame.draw.line(surf, TIER["critical"], (x - 10, y + 10), (x + 10, y - 10), 4)
    if focus:  # the area shown in the pop-up
        (fx, fy), _ = focus
        a, b = view.px(fx - POP_SPAN / 2, fy + POP_SPAN / 2), view.px(fx + POP_SPAN / 2, fy - POP_SPAN / 2)
        pygame.draw.rect(surf, INK, pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]), 2)


def draw_popup(screen, session, frame, fonts, focus, pinned, overview_view) -> pygame.Rect:
    (fx, fy), zone_name = focus
    z = frame.zones.get(zone_name) if zone_name else None
    tier = z.tier if z else "none"
    # put the pop-up in the corner furthest from the hotspot
    hx, hy = overview_view.px(fx, fy)
    px_ = 16 if hx > OW / 2 else OW - POP - 16
    py_ = 16 if hy > OH / 2 else OH - POP - 16
    rect = pygame.Rect(px_, py_, POP, POP)
    pygame.draw.line(screen, INK, (hx, hy), rect.center, 2)  # leader line from hotspot to pop-up

    surf = pygame.Surface((POP, POP))
    v = View(fx - POP_SPAN / 2, fy - POP_SPAN / 2, POP_SPAN, POP_SPAN, POP, POP)
    draw_floor(surf, v, fonts, labels=False)
    draw_paths(surf, v, session, frame, only_pair=z.pair if z and z.pair else None)
    draw_agents(surf, v, frame, fonts, detail=True)
    screen.fill(TIER[tier], rect.inflate(10, 10))
    screen.blit(surf, rect.topleft)

    head = pygame.Rect(rect.x, rect.y, POP, 34)
    screen.fill(TIER[tier], head)
    title = f"ZOOM {zone_name or 'point'}" + ("  PINNED" if pinned else "")
    if z and z.tier != "none":
        if z.ttc <= 0.05:
            title += f"  ·  {z.tier.upper()}  ·  CONTACT NOW"
        elif z.p < 0.3:
            title += f"  ·  {z.tier.upper()}  ·  clearing"
        else:
            ttc = "-" if z.ttc == float("inf") else f"{z.ttc:.1f} s"
            title += f"  ·  {z.tier.upper()}  {z.p:.0%}  ·  contact in {ttc}"
    screen.blit(fonts["med_b"].render(title, True, WHITE), (rect.x + 10, rect.y + 8))
    close = pygame.Rect(rect.right - 30, rect.y + 5, 24, 24)
    screen.blit(fonts["med_b"].render("x", True, WHITE), (close.x + 7, close.y + 1))
    if z and z.pair:
        reasons = session.reasons.get(z.pair, [])
        foot = f"{z.pair[0]} vs {z.pair[1]}" + (f"  ·  fix: {reasons[0][0]}" if reasons and reasons[0][1] > 0 else "")
        fr = pygame.Rect(rect.x, rect.bottom - 28, POP, 28)
        screen.fill(INK, fr)
        screen.blit(fonts["small"].render(foot[:62], True, WHITE), (fr.x + 10, fr.y + 7))
    return close


def draw_panel(screen, session, frame, fonts, whatif, auto, paused, speed) -> None:
    x0 = OW
    screen.fill(WHITE, pygame.Rect(x0, 0, PANEL, OH))
    pygame.draw.line(screen, (226, 232, 240), (x0, 0), (x0, OH), 2)
    hot = session.hottest()
    tier = hot.tier if hot else "none"
    screen.fill(TIER[tier], pygame.Rect(x0, 0, PANEL, 58))
    screen.blit(fonts["big"].render("CLEAR" if tier == "none" else f"{tier.upper()} · {hot.zone}", True, WHITE),
                (x0 + 16, 14))
    st = session.stats()
    y = 70
    screen.blit(fonts["med"].render(f"t = {st['t']:.0f} s   {'PAUSED' if paused else f'x{speed:g}'}   "
                                    f"zoom: {'AUTO' if auto else 'PINNED'}", True, INK), (x0 + 16, y))
    y += 28
    # zone grid (3 x 3), same layout as the floor
    screen.blit(fonts["small"].render("ZONES", True, QUIET), (x0 + 16, y))
    y += 18
    cell = 54
    for i, (row, _) in enumerate(AISLES_Y.items()):
        for j, (col, _) in enumerate(WALKS_X.items()):
            name = f"{row}{col}"
            z = frame.zones[name]
            r = pygame.Rect(x0 + 16 + j * (cell + 8), y + i * (cell // 2 + 6), cell, cell // 2)
            pygame.draw.rect(screen, TIER[z.tier] if z.tier != "none" else (241, 245, 249), r, border_radius=6)
            txt = f"{ZONE_KEYS.index(name) + 1} {name}"
            screen.blit(fonts["small"].render(txt, True, WHITE if z.tier != "none" else QUIET), (r.x + 8, r.y + 6))
    y += 3 * (cell // 2 + 6) + 10
    rows = [("Encounters at corners", st["encounters"], INK), ("Collisions", st["contacts"], TIER["critical"]),
            ("Alerts  advisory / warning / critical", f"{st['advisory']} / {st['warning']} / {st['critical']}", INK)]
    for label, value, colour in rows:
        screen.blit(fonts["med"].render(f"{label}: {value}", True, colour), (x0 + 16, y))
        y += 22
    y += 8
    screen.blit(fonts["small"].render("ALERT LOG", True, QUIET), (x0 + 16, y))
    y += 18
    for a in session.alerts[-8:][::-1]:
        zone = session.alert_zone.get(id(a), "")
        screen.blit(fonts["mono"].render(f"{a.t:5.1f}s {zone} {a.tier:<8} {a.pair[0]}/{a.pair[1]}", True, TIER[a.tier]),
                    (x0 + 16, y))
        y += 18
    y = OH - 92
    for key, label, on in (("A", "alerts reach worker & driver", whatif.workers_react_to_alerts),
                           ("M", "mirrors at every corner", whatif.mirror),
                           ("S", "speed limit 1 m/s at corners", whatif.speed_limit)):
        colour = TIER["none"] if on else QUIET
        screen.blit(fonts["med"].render(f"[{key}] {label}: {'ON' if on else 'off'}", True, colour), (x0 + 16, y))
        y += 24


def draw_bar(screen, fonts) -> None:
    r = pygame.Rect(0, OH, OW + PANEL, BAR)
    screen.fill(INK, r)
    text = ("Click a corner or any point: zoom there   ·   1-9 zones   0 / Esc auto   ·   A alerts   M mirrors   "
            "S speed limit   ·   R new shift   SPACE pause   +/- speed   Q quit")
    screen.blit(fonts["small"].render(text, True, (203, 213, 225)), (16, OH + 14))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--forklifts", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--phone", default="all", help="'all', 'none', or a worker id (e.g. wk_1) whose alerts go to the phone")
    ap.add_argument("--mqtt", action="store_true", help="also publish to phone.html via Mosquitto")
    ap.add_argument("--record", default=None, help="save the window to an .mp4 (e.g. data/warehouse_demo.mp4)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument("--screenshot", default=None, help="save the last frame to this .png (testing)")
    args = ap.parse_args()
    cfg = load_config()

    from iwm.alerts import NtfyNotifier, Publisher

    publisher = Publisher(cfg) if args.phone != "none" and (args.mqtt or NtfyNotifier(cfg).enabled) else None
    last_sent: dict = {}

    pygame.init()
    screen = pygame.display.set_mode((OW + PANEL, OH + BAR))
    pygame.display.set_caption("Industrial World Model - warehouse overview")
    fonts = {"tiny": pygame.font.SysFont("arial", 12), "small": pygame.font.SysFont("arial", 13),
             "med": pygame.font.SysFont("arial", 15), "med_b": pygame.font.SysFont("arial", 15, bold=True),
             "big": pygame.font.SysFont("arial", 26, bold=True),
             "mono": pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 13)}
    clock = pygame.time.Clock()
    recorder = None
    if args.record:
        from iwm.video import VideoRecorder

        recorder = VideoRecorder(args.record, fps=30)

    whatif, seed = WhatIf(), args.seed
    session = WarehouseSession(cfg, seed, whatif, args.forklifts, args.workers)
    frame = session.step()
    overview = View(0, 0, WIDTH, HEIGHT, OW, OH)
    paused, acc, speed = False, 0.0, args.speed
    auto, pinned_focus = True, None  # pinned_focus = ((x, y), zone or None)
    auto_focus, auto_until = None, 0.0
    close_btn, contacts_shown, n = None, [], 0

    running = True
    while running:
        dt_real = clock.tick(30) / 1000.0
        restart = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if close_btn and close_btn.collidepoint(ev.pos):
                    auto, pinned_focus = True, None
                elif ev.pos[0] < OW and ev.pos[1] < OH:
                    x, y = overview.to_floor(*ev.pos)
                    near = min(ZONES.items(), key=lambda kv: np.hypot(kv[1][0] - x, kv[1][1] - y))
                    if np.hypot(near[1][0] - x, near[1][1] - y) < 3.0:
                        pinned_focus = (near[1], near[0])
                    else:
                        pinned_focus = ((x, y), None)
                    auto = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_0:
                    auto, pinned_focus = True, None
                elif pygame.K_1 <= ev.key <= pygame.K_9:
                    name = ZONE_KEYS[ev.key - pygame.K_1]
                    auto, pinned_focus = False, (ZONES[name], name)
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(4.0, speed * 1.5)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(0.25, speed / 1.5)
                elif ev.key == pygame.K_r:
                    seed, restart = random.randint(0, 10_000), True
                elif ev.key == pygame.K_a:
                    whatif.workers_react_to_alerts, restart = not whatif.workers_react_to_alerts, True
                elif ev.key == pygame.K_m:
                    whatif.mirror, restart = not whatif.mirror, True
                elif ev.key == pygame.K_s:
                    whatif.speed_limit, restart = not whatif.speed_limit, True
        if restart:
            session = WarehouseSession(cfg, seed, whatif, args.forklifts, args.workers)
            frame, auto_focus, contacts_shown = session.step(), None, []

        if not paused:
            acc += dt_real * speed
            while acc >= cfg["sim"]["dt"]:
                acc -= cfg["sim"]["dt"]
                frame = session.step()
                contacts_shown = [c for c in contacts_shown if frame.t - c.t < 2.0] + frame.new_contacts
                for a in frame.new_alerts:
                    send = publisher and a.tier in ("warning", "critical") and args.phone in ("all", a.pair[1])
                    if send and time.time() - last_sent.get(tuple(a.pair), 0) > 4:  # no spamming the phone
                        publisher.send(a)
                        last_sent[tuple(a.pair)] = time.time()

        # auto zoom: follow the hottest corner; hold it a few seconds after it clears
        hot = session.hottest()
        now = frame.t
        if hot and TIERS.index(hot.tier) >= TIERS.index("advisory"):
            current = frame.zones[auto_focus[1]] if auto_focus and auto_focus[1] else None
            if (current is None or current.tier == "none"
                    or (TIERS.index(hot.tier), hot.p) > (TIERS.index(current.tier), current.p) + (1, 0)):
                auto_focus = (ZONES[hot.zone], hot.zone)
            auto_until = now + HOLD_S
        elif auto_focus and now > auto_until:
            auto_focus = None
        focus = pinned_focus if not auto else auto_focus

        overview_surf = pygame.Surface((OW, OH))
        draw_overview(overview_surf, overview, session, frame, fonts, focus, contacts_shown)
        screen.blit(overview_surf, (0, 0))
        close_btn = draw_popup(screen, session, frame, fonts, focus, not auto, overview) if focus else None
        draw_panel(screen, session, frame, fonts, whatif, auto, paused, speed)
        draw_bar(screen, fonts)
        pygame.display.flip()
        if recorder:
            rgb = pygame.surfarray.array3d(screen).swapaxes(0, 1)
            recorder.write(rgb[:, :, ::-1].copy())
        n += 1
        if args.frames and n >= args.frames:
            running = False
    if args.screenshot:
        pygame.image.save(screen, args.screenshot)
    if recorder:
        recorder.close()
    pygame.quit()


if __name__ == "__main__":
    main()
