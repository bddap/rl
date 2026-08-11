#!/usr/bin/env python3
"""Chord Atlas — animated design mock for rl#358 (proposal: zoom-fractal quadrants).

Renders the REAL GCR_CHORDS registry (combos.json, extracted from net/src/controls.rs)
as a self-similar diamond space: every node has four child quadrants (d-pad dirs),
each press dives one level (the child zooms to fill the screen), the opposite press
surfaces. Commands are rooms at their code's address. A locked family renders as
teased fog and unfurls on unlock (owner amendment). Each press is a note (rl#359):
the HUD ribbon shows the entered code as a pitch contour, and each room carries its
tune as a sparkline.

Output: frames/*.png (assemble with ffmpeg palettegen/paletteuse).
"""
import json
import math
from glob import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W = H = 800
FPS = 12

COMBOS = json.load(open(HERE / "combos.json"))

# ---- geometry: the self-similar quadrant space ---------------------------------
DIRV = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
STEP = 0.62   # child center offset, in parent radii
SHRINK = 0.36  # child radius, in parent radii


def node_geom(code):
    """(x, y, r) of a code's node in world space."""
    x, y, r = 0.0, 0.0, 1.0
    for c in code:
        dx, dy = DIRV[c]
        x += dx * r * STEP
        y += dy * r * STEP
        r *= SHRINK
    return x, y, r


# Tree: every prefix of every registered code.
NODES = {""}
for e in COMBOS:
    for i in range(1, len(e["code"]) + 1):
        NODES.add(e["code"][:i])
TERMINAL = {e["code"]: e for e in COMBOS}
LOCKED_PREFIX = "DU"  # the night-bloom family plays the locked/unlock story


def in_locked(code):
    return code.startswith(LOCKED_PREFIX) and code != LOCKED_PREFIX

# ---- melody (rl#359): one note per direction, pitch echoes the vertical axis ----
NOTE = {"U": "G", "R": "E", "L": "C", "D": "A"}
PITCH = {"U": 0.0, "R": 0.35, "L": 0.65, "D": 1.0}  # 0 = high

# ---- palette -------------------------------------------------------------------
BG = (13, 16, 22)
INK = (232, 234, 237)
DIM = (120, 128, 140)
FAMILY = {"U": (111, 177, 232), "D": (217, 152, 95), "L": (123, 201, 138), "R": (217, 130, 181)}
FOG = (138, 127, 168)
GOLD = (240, 200, 110)


def fam_color(code):
    return FAMILY[code[0]] if code else INK


def with_a(rgb, a):
    return (*rgb, max(0, min(255, int(a))))


FONT_PATH = sorted(glob("/nix/store/*dejavu-fonts*/share/fonts/truetype/DejaVuSans.ttf"))[-1]
FONT_BOLD = sorted(glob("/nix/store/*dejavu-fonts*/share/fonts/truetype/DejaVuSans-Bold.ttf"))[-1]
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)
    return _fonts[key]


def smooth(t):
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t

# ---- camera --------------------------------------------------------------------
ROOT_ZOOM = 330.0


def cam_root():
    return 0.0, 0.0, ROOT_ZOOM


DPAD_GLYPH = {"U": "^", "D": "v", "L": "<", "R": ">"}


def draw_tune(d, cx, cy, w, code, color, alpha=255):
    """Pitch-contour sparkline of a code — its tune."""
    if not code:
        return
    n = len(code)
    pts = []
    for i, c in enumerate(code):
        px = cx - w / 2 + (w * (i + 0.5) / n)
        py = cy + (PITCH[c] - 0.5) * 14
        pts.append((px, py))
    if len(pts) > 1:
        d.line(pts, fill=with_a(color, alpha * 0.7), width=2)
    for p in pts:
        d.ellipse([p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5], fill=with_a(color, alpha))


def draw_scene(path, frame_idx, cam, pressed=None, fog=1.0, unfurl=1.0, flash=None,
               caption="", title=None, ribbon="", highlight=None, endcard=None):
    img = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    cx, cy, zoom = cam

    def to_screen(x, y):
        return (x - cx) * zoom + W / 2, (y - cy) * zoom + H / 2

    # Nodes, shallow-first so deep rings draw over ancestor ghosts.
    for code in sorted(NODES, key=len):
        locked_member = in_locked(code)
        if locked_member and fog >= 1.0:
            continue  # hidden behind the fog until the unlock starts
        x, y, r = node_geom(code)
        scale = 1.0
        if locked_member and unfurl < 1.0:
            scale = max(0.0, unfurl * 1.15 - 0.15 * unfurl * unfurl)  # slight overshoot ease
        sx, sy = to_screen(x, y)
        sr = r * zoom * scale
        if sr < 2.5 or sx < -sr - 50 or sx > W + sr + 50 or sy < -sr - 50 or sy > H + sr + 50:
            continue
        col = fam_color(code)
        is_room = code in TERMINAL
        ring_a = 200 if sr < 240 else 60  # huge ancestor rings ghost at the edges
        if code == "":
            ring_a = 70
        d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                  outline=with_a(col, ring_a), width=max(1, int(min(4, sr / 40))))
        if is_room:
            e = TERMINAL[code]
            hi = highlight == code
            fill_a = 235 if hi else 130
            rr = sr * 0.28 if sr > 40 else min(sr * 0.5, 6)
            d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=with_a(col, fill_a))
            if hi:
                pr = rr + 6 + 3 * math.sin(frame_idx * 0.5)
                d.ellipse([sx - pr, sy - pr, sx + pr, sy + pr], outline=with_a(GOLD, 220), width=3)
            if sr > 36:
                fs = max(11, min(20, int(sr / 7)))
                label = e["label"]
                tw = d.textlength(label, font=font(fs, bold=hi))
                d.text((sx - tw / 2, sy + rr + 4), label, font=font(fs, bold=hi),
                       fill=with_a(INK if hi else DIM, 255 if hi else 220))
                # the room's tune — the code as pitch contour (rl#359)
                draw_tune(d, sx, sy + rr + 4 + fs + 12, min(sr * 0.8, 12 * len(code)),
                          code, col, 230)
        else:
            # waypoint: the direction glyph + its note letter
            if 12 < sr < 300 and code:
                g = DPAD_GLYPH[code[-1]]
                fs = max(12, min(26, int(sr / 5)))
                d.text((sx - fs / 3, sy - fs * 0.65), g, font=font(fs, bold=True),
                       fill=with_a(col, 190))
                if sr > 55:
                    d.text((sx - fs / 3 + 1, sy + fs * 0.25), NOTE[code[-1]],
                           font=font(int(fs * 0.55)), fill=with_a(DIM, 170))
        # unclaimed quadrants: faint seeds — the space keeps going
        if not locked_member and sr > 110 and len(code) < 8:
            for c2 in "UDLR":
                child = code + c2
                if child in NODES or (child.startswith(LOCKED_PREFIX) and child != LOCKED_PREFIX):
                    continue
                dx, dy = DIRV[c2]
                gx, gy = to_screen(x + dx * r * STEP, y + dy * r * STEP)
                gr = sr * SHRINK
                d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=with_a(DIM, 26), width=1)

    # Fog over the locked family: teased, not hidden — six seeds glint inside.
    if fog > 0.0:
        fx, fy, fr = node_geom(LOCKED_PREFIX)
        sx, sy = to_screen(fx, fy)
        sr = fr * zoom * 1.05
        if -sr < sx < W + sr and -sr < sy < H + sr and sr > 4:
            for k in range(4):
                a = fog * (70 - k * 14)
                rr = sr * (0.55 + 0.16 * k)
                d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=with_a(FOG, a))
            # teased: one faint seed per hidden room, arranged where they truly live
            for code in TERMINAL:
                if not in_locked(code):
                    continue
                nx, ny, _ = node_geom(code)
                px, py = to_screen(nx, ny)
                d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=with_a(INK, fog * 90))
            if sr > 30:
                fs = max(12, min(30, int(sr / 5)))
                d.text((sx - fs * 0.3, sy - fs * 0.6), "?", font=font(fs, bold=True),
                       fill=with_a(INK, fog * 160))
                if sr > 90:
                    t = "locked — 6 codes sleep here"
                    tw = d.textlength(t, font=font(13))
                    d.text((sx - tw / 2, sy + sr * 0.62), t, font=font(13),
                           fill=with_a(INK, fog * 150))

    # Unlock flash ring.
    if flash is not None:
        fx, fy, fr = node_geom(LOCKED_PREFIX)
        sx, sy = to_screen(fx, fy)
        rr = fr * zoom * (0.6 + 2.6 * flash)
        d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr],
                  outline=with_a(GOLD, 240 * (1 - flash)), width=4)

    # ---- HUD -------------------------------------------------------------------
    # melody ribbon: the entered code as pitch contour + glyphs (rl#359)
    if ribbon:
        d.rounded_rectangle([16, 16, 36 + 30 * len(ribbon), 92], 10, fill=(0, 0, 0, 150))
        for i, c in enumerate(ribbon):
            px = 36 + 30 * i
            py = 34 + PITCH[c] * 26
            if i:
                pc = ribbon[i - 1]
                d.line([36 + 30 * (i - 1), 34 + PITCH[pc] * 26, px, py],
                       fill=with_a(fam_color(ribbon[0]), 170), width=2)
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=with_a(fam_color(c), 255))
            d.text((px - 4, 66), DPAD_GLYPH[c], font=font(14, bold=True), fill=with_a(INK, 220))
        d.text((22, 20), "·".join(NOTE[c] for c in ribbon), font=font(12), fill=with_a(DIM, 200))

    # d-pad indicator, bottom-left
    px, py, s = 60, H - 64, 16
    for c, (dx, dy) in DIRV.items():
        bx, by = px + dx * s * 1.4, py + dy * s * 1.4
        on = pressed == c
        d.rounded_rectangle([bx - s * 0.62, by - s * 0.62, bx + s * 0.62, by + s * 0.62], 4,
                            fill=with_a(fam_color(c) if on else (60, 66, 76), 255 if on else 170))
    d.rounded_rectangle([px - s * 0.62, py - s * 0.62, px + s * 0.62, py + s * 0.62], 4,
                        fill=with_a((60, 66, 76), 170))

    if caption:
        tw = d.textlength(caption, font=font(17))
        d.rounded_rectangle([W / 2 - tw / 2 - 14, H - 58, W / 2 + tw / 2 + 14, H - 26], 8,
                            fill=(0, 0, 0, 160))
        d.text((W / 2 - tw / 2, H - 52), caption, font=font(17), fill=with_a(INK, 240))

    if title:
        t1, t2 = title
        tw = d.textlength(t1, font=font(34, bold=True))
        d.text((W / 2 - tw / 2, 96), t1, font=font(34, bold=True), fill=with_a(INK, 245))
        tw = d.textlength(t2, font=font(16))
        d.text((W / 2 - tw / 2, 142), t2, font=font(16), fill=with_a(DIM, 235))

    img.paste(ov, (0, 0), ov)

    if endcard:
        # separate overlay: ImageDraw REPLACES pixels, so a translucent rect drawn on
        # the scene overlay would erase the atlas instead of dimming it
        ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(ov2)
        d2.rectangle([0, 0, W, H], fill=(0, 0, 0, 110))
        y0 = 260
        for i, line in enumerate(endcard):
            f = font(26 if i == 0 else 17, bold=i == 0)
            tw = d2.textlength(line, font=f)
            d2.text((W / 2 - tw / 2, y0), line, font=f, fill=with_a(INK, 250))
            y0 += 46 if i == 0 else 30
        img.paste(ov2, (0, 0), ov2)
    img.save(path)


# ---- storyboard ----------------------------------------------------------------
frames_dir = HERE / "frames"
frames_dir.mkdir(exist_ok=True)
FRAME = 0


def emit(**kw):
    global FRAME
    draw_scene(frames_dir / f"f{FRAME:04d}.png", FRAME, **kw)
    FRAME += 1


def dive(frm, to, n, pressed, fog, ribbon_after, caption, unfurl=1.0, highlight=None):
    (x0, y0, z0), (x1, y1, z1) = frm, to
    for i in range(n):
        t = smooth((i + 1) / n)
        cam = (lerp(x0, x1, t), lerp(y0, y1, t), z0 * (z1 / z0) ** t)
        emit(cam=cam, pressed=pressed if i < n // 2 else None, fog=fog, unfurl=unfurl,
             ribbon=ribbon_after, caption=caption, highlight=highlight)


def hold(cam, n, fog=1.0, **kw):
    for _ in range(n):
        emit(cam=cam, fog=fog, **kw)


C_ROOT = cam_root()


def C(code):
    x, y, r = node_geom(code)
    return x, y, ROOT_ZOOM / r


# 1 — the atlas at rest
hold(C_ROOT, 30, title=("CHORD ATLAS", "hold X — the code space is a place; each press is a step and a note"),
     caption="24 commands live at addresses in an infinite 4-ary space", ribbon="")

# 2 — dive Up (note G)
dive(C_ROOT, C("U"), 16, "U", 1.0, "U", "press ^  — dive into the sky wing (note: G)")
hold(C("U"), 10, ribbon="U", caption="vehicles live up here — opposite press surfaces back out")

# 3 — dive Left → Enter plane
dive(C("U"), C("UL"), 16, "L", 1.0, "UL", "press <  — arrive: Enter plane (tune: G·C)", highlight="UL")
hold(C("UL"), 16, ribbon="UL", caption="^<  Enter plane — every code is a room, every room a tune",
     highlight="UL")

# 4 — surface back to root (opposite presses)
dive(C("UL"), C("U"), 10, "R", 1.0, "U", "press >  — surface")
dive(C("U"), C_ROOT, 10, "D", 1.0, "", "press v  — surface to the atlas")

# 5 — dive Down: the ground wing, bloom family fogged
dive(C_ROOT, C("D"), 16, "D", 1.0, "D", "press v  — the ground wing (note: A)")
hold(C("D"), 20, ribbon="D", caption="locked space is TEASED: six seeds sleep behind the fog")

# 6 — the unlock: fog dissolves, the subtree unfurls
for i in range(26):
    t = (i + 1) / 26
    emit(cam=C("D"), fog=max(0.0, 1 - t * 1.6), unfurl=smooth(min(1, t * 1.25)),
         flash=t, ribbon="D", caption="UNLOCK — the atlas GROWS: bloom family unfurls (6 new tunes)")
hold(C("D"), 12, fog=0.0, ribbon="D",
     caption="UNLOCK — the atlas GROWS: bloom family unfurls (6 new tunes)")

# 7 — dive into the new family, then one more level to walk among the rooms
dive(C("D"), C("DU"), 16, "U", 0.0, "DU", "press ^  — walk the new wing (tune: A·G)")
hold(C("DU"), 10, fog=0.0, ribbon="DU", caption="press ^  — walk the new wing (tune: A·G)")
dive(C("DU"), C("DUU"), 14, "U", 0.0, "DUU", "press ^  — deeper (tune: A·G·G)")
hold(C("DUU"), 26, fog=0.0, ribbon="DUU",
     caption="new rooms, each with its own melody — pitch contours ARE the signposts")

# 8 — end card over the full atlas
dive(C("DUU"), C_ROOT, 16, None, 0.0, "", "")
hold(C_ROOT, 34, fog=0.0, endcard=[
    "CHORD ATLAS",
    "the combo map as a self-similar place: press = dive, opposite = surface",
    "commands are rooms; locked wings are teased fog that unfurls on unlock",
    "every press is a note (rl#359) — the path you walk is the tune you play",
    "rendered from the real GCR_CHORDS registry (24 codes)",
])

print(f"{FRAME} frames")
