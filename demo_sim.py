"""Live top-down demo in pygame: the world model watching a blind corner.

    python demo_sim.py                 # blind-corner demo
    python demo_sim.py --mqtt          # also buzz the phone (needs mosquitto + phone.html)

Keys:
    SPACE pause      R  new random crossing      B  blind-corner demo again
    A  workers/driver react to our alerts (on/off)
    M  convex mirror (on/off)      S  1 m/s speed limit near the corner (on/off)
    Q / Esc quit

What you see: racking around a blind corner, the forklift (orange) and the
walker (blue), their predicted paths for the next 5 s, the risk, the alert tier,
the top reason, and when a plain "detection-only" camera would have fired.
"""

import argparse
import random

import pygame

from iwm import load_config
from iwm.demo import DemoSession
from iwm.sim import WhatIf

PX = 50  # pixels per metre
PANEL = 360
BG, RACK, AISLE, INK, QUIET = (241, 245, 249), (148, 163, 184), (226, 232, 240), (15, 23, 42), (100, 116, 139)
FORKLIFT, WORKER = (234, 88, 12), (37, 99, 235)
TIER = {"none": (22, 163, 74), "advisory": (202, 138, 4), "warning": (234, 88, 12), "critical": (220, 38, 38)}


def to_px(x, y, H):
    return int(x * PX), int((H - y) * PX)  # y up in metres, down on screen


def draw_floor(screen, cfg):
    s = cfg["sim"]
    W, H = s["width_m"], s["height_m"]
    cx, cy = s["corner"]
    screen.fill(RACK, pygame.Rect(0, 0, W * PX, H * PX))
    # forklift aisle (horizontal, 3 m wide) and walkway (vertical, 2 m wide)
    pygame.draw.rect(screen, AISLE, pygame.Rect(0, (H - cy - 1.5) * PX, W * PX, 3 * PX))
    pygame.draw.rect(screen, AISLE, pygame.Rect((cx - 1) * PX, 0, 2 * PX, H * PX))
    # walkway markings
    for y in range(0, int(H * PX), 24):
        pygame.draw.line(screen, (250, 204, 21), ((cx - 1) * PX, y), ((cx - 1) * PX, y + 12), 3)
        pygame.draw.line(screen, (250, 204, 21), ((cx + 1) * PX, y), ((cx + 1) * PX, y + 12), 3)


def draw_frame(screen, fonts, cfg, session, frame, whatif, paused, flash):
    s = cfg["sim"]
    H = s["height_m"]
    draw_floor(screen, cfg)
    f_small, f_med, f_big = fonts

    # predicted paths
    for r in frame.risks:
        colour = TIER[session.engine.tier_of(r.forklift, r.worker)]
        for path in r.paths.values():
            pts = [to_px(x, y, H) for x, y in path]
            if len(pts) > 1:
                pygame.draw.lines(screen, colour, False, pts, 3)
                pygame.draw.circle(screen, colour, pts[-1], 5)

    # agents
    for tr in frame.tracks:
        x, y = to_px(tr.x, tr.y, H)
        if tr.type == "forklift":
            pygame.draw.rect(screen, FORKLIFT, pygame.Rect(x - 28, y - 16, 56, 32), border_radius=6)
            label = "Forklift"
        else:
            pygame.draw.circle(screen, WORKER, (x, y), 14)
            label = "Worker"
        screen.blit(f_small.render(f"{label} {((tr.vx ** 2 + tr.vy ** 2) ** 0.5):.1f} m/s", True, INK), (x - 30, y - 38))

    if frame.contact:
        cx, cy = to_px(*s["corner"], H)
        screen.blit(f_big.render("CONTACT", True, TIER["critical"]), (cx - 80, cy - 120))

    # side panel
    W = int(s["width_m"] * PX)
    panel = pygame.Rect(W, 0, PANEL, int(H * PX))
    screen.fill((255, 255, 255), panel)
    pygame.draw.rect(screen, TIER[frame.tier], pygame.Rect(W, 0, PANEL, 70))
    screen.blit(f_big.render(frame.tier.upper() if frame.tier != "none" else "CLEAR", True, (255, 255, 255)), (W + 16, 18))
    top = max(frame.risks, key=lambda r: r.p_collision, default=None)
    y = 86
    lines = [(f"t = {frame.t:4.1f} s", INK)]
    if top:
        ttc = "-" if top.ttc_s == float("inf") else f"{top.ttc_s:.1f} s"
        lines += [(f"Collision risk  {top.p_collision * 100:3.0f}%", INK), (f"Time to contact {ttc}", INK),
                  (f"Distance now    {top.distance_m:.1f} m", INK)]
    lines += [("", INK), ("Detection-only camera:", QUIET),
              ("  FIRING" if frame.detection_only else "  silent", TIER["critical"] if frame.detection_only else QUIET)]
    if session.alerts:
        lines += [("", INK), ("Why (top fix):", QUIET), (f"  {session.alerts[-1].cause[:38]}", INK)]
    lines += [("", INK), ("Alert log:", QUIET)]
    lines += [(f"  {a.t:4.1f}s {a.tier:<8} p={a.p_collision:.2f}", TIER[a.tier]) for a in session.alerts[-5:]]
    lines += [("", INK), ("What-if  [A] alerts react: " + ("ON" if whatif.workers_react_to_alerts else "off"), QUIET),
              ("         [M] mirror: " + ("ON" if whatif.mirror else "off"), QUIET),
              ("         [S] speed limit: " + ("ON" if whatif.speed_limit else "off"), QUIET),
              ("[B] demo  [R] random  [SPACE] " + ("resume" if paused else "pause"), QUIET)]
    if session.done:
        sm = session.summary()
        lines += [("", INK), (f"Contact: {sm['contact_s']} s   closest {sm['closest_m']} m", INK),
                  (f"Warned {sm['lead_first_alert_s']} s early vs {sm['lead_detection_only_s']} s", INK)]
    for text, colour in lines:
        screen.blit(f_med.render(text, True, colour), (W + 16, y))
        y += 22

    if flash:
        pygame.draw.rect(screen, TIER["critical"], pygame.Rect(0, 0, W, int(H * PX)), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt", action="store_true", help="publish alerts so phone.html buzzes")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    args = ap.parse_args()
    cfg = load_config()
    publisher = None
    if args.mqtt:
        from iwm.alerts import Publisher

        publisher = Publisher(cfg)

    pygame.init()
    s = cfg["sim"]
    screen = pygame.display.set_mode((int(s["width_m"] * PX) + PANEL, int(s["height_m"] * PX)))
    pygame.display.set_caption("Industrial World Model - live demo")
    fonts = (pygame.font.SysFont("arial", 14), pygame.font.SysFont("consolas,menlo,monospace", 16),
             pygame.font.SysFont("arial", 30, bold=True))
    clock = pygame.time.Clock()

    whatif, scenario, seed = WhatIf(), "blind_corner", 1
    session = DemoSession(cfg, scenario, seed, whatif)
    frame = session.step()
    paused, acc, flash_until = False, 0.0, 0.0

    running = True
    while running:
        dt_real = clock.tick(30) / 1000.0
        restart = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_b:
                    scenario, seed, restart = "blind_corner", 1, True
                elif ev.key == pygame.K_r:
                    scenario, seed, restart = "crossing", random.randint(0, 10_000), True
                elif ev.key == pygame.K_a:
                    whatif.workers_react_to_alerts, restart = not whatif.workers_react_to_alerts, True
                elif ev.key == pygame.K_m:
                    whatif.mirror, restart = not whatif.mirror, True
                elif ev.key == pygame.K_s:
                    whatif.speed_limit, restart = not whatif.speed_limit, True
        if restart:
            session = DemoSession(cfg, scenario, seed, whatif)
            frame, paused = session.step(), False

        if not paused and not session.done:
            acc += dt_real * args.speed
            while acc >= cfg["sim"]["dt"] and not session.done:
                acc -= cfg["sim"]["dt"]
                frame = session.step()
                for a in frame.new_alerts:
                    if publisher:
                        publisher.send(a)
                    if a.tier in ("warning", "critical"):
                        flash_until = pygame.time.get_ticks() + 400

        draw_frame(screen, fonts, cfg, session, frame, whatif, paused, pygame.time.get_ticks() < flash_until)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
