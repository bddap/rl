#!/usr/bin/env python3
"""Design mock for rl#358: THE HYPERBOLIC ATLAS.

Renders the real GCR_CHORDS registry (net/src/controls.rs) as a Poincare-disk
hyperbolic tree and animates: d-pad navigation as hyperbolic glides, district
landmarks (place memory), pitch glyphs per press (rl#359), and an unlock that
blooms a fogged region open.

Run:  nix-shell -p python3 python3Packages.pillow python3Packages.numpy \
        --run "python3 generate.py"
Emits dpad-map-hyperatlas.gif plus a few still PNGs next to this file.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/nix/store/238xkvzxd8wpwxiiffslzi5kfdsql6wj-dejavu-fonts-2.37/share/fonts/truetype"


def font(sz, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), sz)


# ---------------------------------------------------------------- real data
U, D, L, R = "^", "v", "<", ">"
# Verbatim from net/src/controls.rs GCR_CHORDS (24 entries).
CHORDS = [
    (U + L, "Enter plane"), (U + R, "Enter ship"), (D + D, "Exit to foot"),
    (U + U + U, "Render: mesh"), (U + U + D, "Render: mesh+colliders"),
    (U + U + L, "Render: colliders"),
    (D + U + U + U, "Ground: bloom"), (D + U + U + D, "Ground: bloom aurora"),
    (D + U + U + L, "Ground: bloom ember"), (D + U + U + R, "Ground: bloom frost"),
    (D + U + D + U, "Ground: bloom rose"), (D + U + D + D, "Ground: bloom filigree"),
    (D + L + U, "Ground: shipped"), (D + L + D, "Ground: patterned"),
    (D + L + L, "Ground: wind-combed"), (D + L + R, "Ground: cracked loam"),
    (D + R + U, "Ground: wet nocturne"), (D + R + D, "Ground: watershed"),
    (D + R + L, "Ground: watershed naturalist"), (D + R + R, "Ground: watershed nocturne"),
    (L, "Swap brain"), (R, "Restart round"), (L + R, "Perf overlay"),
    (U + U + D + D + L + R, "Quit round"),
]

DISTRICTS = [  # (prefix, name, rgb) — longest prefix wins
    (U + U, "RENDER OBSERVATORY", (150, 110, 235)),
    (U, "SKY HARBOR", (95, 150, 225)),
    (D + U, "NIGHT-BLOOM GARDEN", (230, 100, 190)),
    (D + L, "LOAM QUARTER", (205, 150, 70)),
    (D + R, "THE WATERSHED", (60, 200, 190)),
    (D, "", (150, 140, 120)),
    (L, "WEST MONOLITH", (140, 150, 160)),
    (R, "EAST MONOLITH", (140, 150, 160)),
]
LOCKED_PREFIX = D + R  # watershed starts locked; unlocking it is the demo

# rl#359: every press sounds a note. Pitch rises with "upness" of the glyph.
NOTE = {D: ("C", 0), L: ("D", 1), R: ("E", 2), U: ("G", 3)}
UNLOCK_NOTE = ("A", 4)  # the unlock extends the scale

DIR_ANGLE = {R: 0.0, U: 90.0, L: 180.0, D: 270.0}  # math coords, y-up


def district_of(code):
    for pre, name, col in DISTRICTS:
        if code.startswith(pre):
            return pre, name, col
    return "", "", (170, 170, 170)


# ------------------------------------------------------------- mobius tools
def m_trans(t):  # translate hyperbolic distance t along +x
    return np.array([[math.cosh(t / 2), math.sinh(t / 2)],
                     [math.sinh(t / 2), math.cosh(t / 2)]], dtype=complex)


def m_rot(theta):
    e = complex(math.cos(theta / 2), math.sin(theta / 2))
    return np.array([[e, 0], [0, e.conjugate()]], dtype=complex)


def m_to_zero(p):  # mobius sending p -> 0
    return np.array([[1, -p], [-p.conjugate(), 1]], dtype=complex)


def m_apply(M, z):
    a, b = M[0]
    c, d = M[1]
    return (a * z + b) / (c * z + d)


# ------------------------------------------------------------------ layout
class Node:
    def __init__(self, code):
        self.code = code
        self.children = {}
        self.label = None
        self.pos = 0j
        self.bearing = 90.0


def build_tree():
    root = Node("")
    for code, label in CHORDS:
        n = root
        for ch in code:
            n = n.children.setdefault(ch, Node(n.code + ch))
        n.label = label
    return root


EDGE_LEN = 1.35


def layout(node, M, bearing):
    node.pos = m_apply(M, 0j)
    node.bearing = bearing
    for d, child in node.children.items():
        if node.code == "":
            b = DIR_ANGLE[d]
        else:
            delta = (DIR_ANGLE[d] - bearing + 180) % 360 - 180
            if abs(abs(delta) - 180) < 1e-6:
                delta = 180.0
            b = bearing + delta * 0.72  # compress so reversal misses the parent
        th = math.radians(b)
        Mc = M @ m_rot(th) @ m_trans(EDGE_LEN) @ m_rot(-th)
        layout(child, Mc, b)


def flatten(node, out):
    out.append(node)
    for c in node.children.values():
        flatten(c, out)
    return out


# ------------------------------------------------------------------ render
W, H = 760, 900
MAP_H = 760
CX, CY, RAD = W // 2, MAP_H // 2, 350
BG = (13, 16, 22)


def to_px(z):
    return CX + z.real * RAD, CY - z.imag * RAD


def conformal(z):
    return max(0.0, 1.0 - abs(z) ** 2)


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class Scene:
    def __init__(self):
        self.root = build_tree()
        layout(self.root, np.eye(2, dtype=complex), 90.0)
        self.nodes = flatten(self.root, [])
        self.by_code = {n.code: n for n in self.nodes}
        self.view = np.eye(2, dtype=complex)
        self.trail = []          # codes of visited path nodes, in order
        self.melody = []         # (letter, rung, color)
        self.unlock_t = 0.0      # 0 locked .. 1 unlocked
        self.unlocking = False
        self.caption = ""
        self.sub = ""
        self.pressed = None      # dir glyph currently pressed
        self.pulse_code = None
        self.pulse_t = 0.0
        self.frames = []

    # -- state helpers
    def vpos(self, node):
        return m_apply(self.view, node.pos)

    def locked(self, node):
        return node.code.startswith(LOCKED_PREFIX) and self.unlock_t < 1.0

    def glide_to(self, code, nframes, k=0.24):
        target = self.by_code[code]
        for _ in range(nframes):
            q = self.vpos(target)
            self.view = m_to_zero(q * k) @ self.view
            self.frame()

    # -- drawing
    def frame(self):
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img, "RGBA")
        # disk
        dr.ellipse([CX - RAD, CY - RAD, CX + RAD, CY + RAD], fill=(18, 22, 31),
                   outline=(60, 70, 95), width=2)
        self.draw_districts(img, dr)
        self.draw_edges(dr)
        self.draw_nodes(dr)
        self.draw_hud(dr)
        self.frames.append(img)

    def draw_districts(self, img, dr):
        blob = Image.new("RGBA", (W, MAP_H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blob)
        for n in self.nodes:
            if n.code == "":
                continue
            pre, name, col = district_of(n.code)
            z = self.vpos(n)
            if abs(z) > 0.985:
                continue
            x, y = to_px(z)
            r = (34 + 8 * len(n.children)) * (0.35 + 0.65 * conformal(z))
            if self.locked(n):
                col = lerp((75, 82, 92), col, self.unlock_t * 0.5)
                a = 26
            else:
                a = 34
            bd.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,))
        blob = blob.filter(ImageFilter.GaussianBlur(14))
        img.paste(blob, (0, 0), blob)
        # district names float over their hub node when visible
        for pre, name, col in DISTRICTS:
            if not name or pre not in self.by_code:
                continue
            n = self.by_code[pre]
            z = self.vpos(n)
            cf = conformal(z)
            locked_tease = self.locked(n)  # a teased lock stays visible further out
            if abs(z) > (0.93 if locked_tease else 0.82) or (cf < 0.25 and not locked_tease):
                continue
            x, y = to_px(z)
            locked = self.locked(n)
            fade = self.unlock_t if pre == LOCKED_PREFIX else 1.0
            colr = (110, 115, 122) if locked else lerp((110, 115, 122), col, max(fade, 0.001))
            txt = "? ? ?" if locked and self.unlock_t < 0.15 else name
            f = font(max(11, int(10 + 9 * cf)), bold=True)
            tw = dr.textlength(txt, font=f)
            dr.text((x - tw / 2, y - 30 - 14 * cf), txt, font=f,
                    fill=colr + (200,))

    def draw_edges(self, dr):
        for n in self.nodes:
            for d, c in n.children.items():
                z1, z2 = self.vpos(n), self.vpos(c)
                if abs(z1) > 0.995 and abs(z2) > 0.995:
                    continue
                x1, y1 = to_px(z1)
                x2, y2 = to_px(z2)
                _, _, col = district_of(c.code)
                lock = self.locked(c)
                on_trail = c.code in self.trail
                if lock:
                    col = lerp((60, 66, 74), col, self.unlock_t)
                if on_trail:
                    col, wdt = (255, 205, 90), 4
                else:
                    wdt = max(1, int(1 + 2.4 * conformal((z1 + z2) / 2)))
                # geodesic approximated with its midpoint (fine at this scale)
                zm = m_apply(m_to_zero(-((z1 + z2) / 2 * 0.28)), 0j)
                dr.line([x1, y1, x2, y2], fill=col + (120 if lock else 220,), width=wdt)
                # pitch glyph: mini 4/5-rung ladder at the edge midpoint
                cf = conformal((z1 + z2) / 2)
                if cf > 0.45 and not lock:
                    letter, rung = NOTE[d]
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    hgt, wid = 13, 9
                    rungs = 5 if self.unlock_t >= 1.0 else 4
                    for i in range(rungs):
                        yy = my + hgt / 2 - i * hgt / (rungs - 1)
                        dr.line([mx - wid / 2, yy, mx + wid / 2, yy],
                                fill=(255, 255, 255, 40), width=1)
                    yy = my + hgt / 2 - rung * hgt / (rungs - 1)
                    dr.ellipse([mx - 3, yy - 3, mx + 3, yy + 3],
                               fill=(255, 225, 140, int(230 * cf)))

    def draw_nodes(self, dr):
        f_small = font(12)
        for n in self.nodes:
            if n.code == "":
                z = self.vpos(n)
                x, y = to_px(z)
                r = 7 * (0.4 + 0.6 * conformal(z))
                dr.ellipse([x - r, y - r, x + r, y + r], outline=(235, 240, 245),
                           width=2, fill=(50, 58, 70))
                continue
            z = self.vpos(n)
            if abs(z) > 0.99:
                continue
            x, y = to_px(z)
            cf = conformal(z)
            _, _, col = district_of(n.code)
            lock = self.locked(n)
            terminal = n.label is not None
            base = 6.5 if terminal else 4.0
            r = base * (0.35 + 0.65 * cf)
            grow = 1.0
            if n.code.startswith(LOCKED_PREFIX) and self.unlocking:
                # staggered bloom: deeper nodes appear later
                depth = len(n.code) - len(LOCKED_PREFIX)
                grow = min(1.0, max(0.0, self.unlock_t * 3.0 - depth * 0.55))
                r *= 0.2 + 1.1 * grow if grow < 0.85 else 1.0
            if lock and not self.unlocking:
                dr.ellipse([x - r, y - r, x + r, y + r], outline=(95, 102, 112, 170),
                           width=1)
                if terminal and cf > 0.5:
                    dr.text((x + 7, y - 7), "?", font=f_small, fill=(95, 102, 112, 200))
                continue
            colr = col if not lock else lerp((95, 102, 112), col, grow)
            if n.code == self.pulse_code and self.pulse_t > 0:
                pr = r + 14 * self.pulse_t
                dr.ellipse([x - pr, y - pr, x + pr, y + pr],
                           outline=colr + (int(200 * self.pulse_t),), width=2)
            if terminal:
                self.landmark(dr, n, x, y, r, colr, cf)
            else:
                dr.ellipse([x - r, y - r, x + r, y + r], fill=colr + (235,))
            if terminal and cf > 0.52 and abs(z) < 0.8:
                a = int(255 * min(1, (cf - 0.5) * 4) * grow)
                if a > 20:
                    dr.text((x + 9, y - 7), n.label, font=f_small,
                            fill=(225, 230, 236, a))

    def landmark(self, dr, n, x, y, r, col, cf):
        """District-specific landmark shapes — place memory, not bullet points."""
        pre, _, _ = district_of(n.code)
        if pre == D + U:  # bloom: 5-petal flower
            for i in range(5):
                a = math.radians(i * 72 - 90)
                px, py = x + math.cos(a) * r, y + math.sin(a) * r
                dr.ellipse([px - r * .55, py - r * .55, px + r * .55, py + r * .55],
                           fill=col + (200,))
            dr.ellipse([x - r * .45, y - r * .45, x + r * .45, y + r * .45],
                       fill=(255, 235, 170, 255))
        elif pre == D + R:  # watershed: ripple arcs
            for k in (1.0, 1.7):
                dr.arc([x - r * k, y - r * k, x + r * k, y + r * k], 200, 340,
                       fill=col + (230,), width=2)
            dr.ellipse([x - r * .5, y - r * .5, x + r * .5, y + r * .5],
                       fill=col + (255,))
        elif pre == D + L:  # loam: ridge triangle
            dr.polygon([x - r, y + r * .8, x, y - r, x + r, y + r * .8],
                       fill=col + (235,))
        elif pre == U + U:  # render observatory: diamond
            dr.polygon([x, y - r * 1.2, x + r, y, x, y + r * 1.2, x - r, y],
                       fill=col + (235,))
        elif pre == U:  # sky harbor: wing chevron
            dr.polygon([x - r, y + r * .6, x, y - r * .8, x + r, y + r * .6,
                        x, y + r * .1], fill=col + (235,))
        else:  # monoliths / exit: plain stele
            dr.rectangle([x - r * .55, y - r, x + r * .55, y + r], fill=col + (235,))

    def draw_hud(self, dr):
        # caption
        if self.caption:
            dr.text((20, 14), self.caption, font=font(20, bold=True),
                    fill=(240, 243, 247))
        if self.sub:
            dr.text((20, 42), self.sub, font=font(14), fill=(160, 170, 185))
        # entered code so far (trail glyphs)
        if self.trail:
            code = self.trail[-1]
            dr.text((20, MAP_H - 34), "code: " + " ".join(code),
                    font=font(17, bold=True), fill=(255, 205, 90))
        # d-pad widget, bottom-left of HUD strip
        cx, cy, s = 74, MAP_H + 70, 22
        for d, (ox, oy) in {U: (0, -1), D: (0, 1), L: (-1, 0), R: (1, 0)}.items():
            x, y = cx + ox * s, cy + oy * s
            fill = (255, 205, 90) if self.pressed == d else (52, 60, 72)
            dr.rounded_rectangle([x - 10, y - 10, x + 10, y + 10], 4, fill=fill)
            glyph = {U: "^", D: "v", L: "<", R: ">"}[d]
            gcol = (20, 20, 25) if self.pressed == d else (150, 160, 175)
            f = font(13, bold=True)
            tw = dr.textlength(glyph, font=f)
            dr.text((x - tw / 2, y - 9), glyph, font=f, fill=gcol)
        dr.rounded_rectangle([cx - 10, cy - 10, cx + 10, cy + 10], 4, fill=(40, 46, 56))
        # staff: rl#359 — every press is a note; the path IS a melody
        sx, sy, sw = 150, MAP_H + 34, W - 190
        rungs = 5 if self.unlock_t >= 1.0 else 4
        labels = ["C", "D", "E", "G", "A"][:rungs]
        for i in range(rungs):
            yy = sy + 80 - i * 20
            new = (i == 4)
            dr.line([sx, yy, sx + sw, yy],
                    fill=(120, 200, 190, 220) if new else (70, 78, 92), width=1)
            dr.text((sx - 16, yy - 8), labels[i], font=font(11),
                    fill=(120, 200, 190) if new else (120, 128, 142))
        for i, (letter, rung, col) in enumerate(self.melody[-14:]):
            xx = sx + 26 + i * 34
            yy = sy + 80 - rung * 20
            dr.ellipse([xx - 7, yy - 7, xx + 7, yy + 7], fill=col + (255,))
            dr.text((xx - 4, yy + 10), letter, font=font(11), fill=(190, 197, 208))
        dr.text((sx, sy - 22), "the code is a melody  (rl#359)", font=font(12),
                fill=(120, 128, 142))

    # -- moves
    def press(self, d, note_col=(255, 205, 90)):
        cur = self.trail[-1] if self.trail else ""
        nxt = cur + d
        self.pressed = d
        letter, rung = NOTE[d]
        self.melody.append((letter, rung, note_col))
        self.trail.append(nxt)
        for _ in range(3):
            self.frame()
        self.glide_to(nxt, 15)
        self.pressed = None

    def pulse(self, code, nframes=16):
        self.pulse_code = code
        for i in range(nframes):
            self.pulse_t = 1.0 - i / nframes
            self.frame()
        self.pulse_code, self.pulse_t = None, 0.0


def main():
    sc = Scene()

    # S0 — overview
    sc.caption = "THE HYPERBOLIC ATLAS"
    sc.sub = "the real 24-chord GCR map in a Poincare disk - every code is a journey"
    for _ in range(22):
        sc.frame()

    # S1 — navigate v ^ ^ v  ->  Ground: bloom aurora
    sc.sub = "hold X, tap d-pad: each press glides you one district deeper"
    _, _, bloom_col = district_of(D + U)
    for d in [D, U, U, D]:
        sc.press(d, note_col=bloom_col)
    sc.caption = "NIGHT-BLOOM GARDEN"
    sc.sub = "v ^ ^ v  ->  Ground: bloom aurora - remembered as a walk, not a list row"
    sc.pulse(D + U + U + D, 18)
    for _ in range(14):
        sc.frame()

    # S2 — release: home
    sc.caption = "release X -> executes, map folds home"
    sc.sub = ""
    sc.trail = []
    sc.glide_to("", 14, k=0.35)

    # S3 — unlock: the watershed blooms open (camera drifts south so the
    # growth happens mid-view, not at the rim)
    sc.caption = "something stirs beyond the loam..."
    sc.sub = ""
    sc.glide_to(D + R, 12, k=0.14)
    still_unlock_pre = len(sc.frames)
    sc.caption = "UNLOCK: THE WATERSHED"
    sc.sub = "progression grows the map - fog lifts, a new region blooms, the scale gains A"
    sc.unlocking = True
    for i in range(34):
        sc.unlock_t = min(1.0, (i + 1) / 26)
        sc.frame()
    sc.melody.append((UNLOCK_NOTE[0], UNLOCK_NOTE[1], (120, 200, 190)))
    for _ in range(12):
        sc.frame()

    # S4 — walk into the new region: v > v -> Ground: watershed
    sc.caption = "NEW GROUND"
    sc.sub = "the unlocked space is real territory - walk in: v > v"
    sc.glide_to("", 12, k=0.3)
    _, _, water_col = district_of(D + R)
    sc.melody = []
    for d in [D, R, D]:
        sc.press(d, note_col=water_col)
    sc.pulse(D + R + D, 18)
    for _ in range(12):
        sc.frame()

    # S5 — fold home, final overview
    sc.caption = "THE HYPERBOLIC ATLAS"
    sc.sub = "landmarks + melody + growth: place memory for a 4-ary code space"
    sc.trail = []
    sc.glide_to("", 16, k=0.3)
    for _ in range(26):
        sc.frame()

    # stills for the issue comment / quick review
    for name, idx in [("still-overview", 20), ("still-garden", 95),
                      ("still-unlock", still_unlock_pre + 24),
                      ("still-final", len(sc.frames) - 2)]:
        sc.frames[idx].save(os.path.join(HERE, f"{name}.png"))

    frames = [f.quantize(colors=128, dither=Image.Dither.NONE) for f in sc.frames]
    out = os.path.join(HERE, "dpad-map-hyperatlas.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=45,
                   loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
