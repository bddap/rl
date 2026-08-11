#!/usr/bin/env python3
"""Animated mock of the hyperbolic-recentering d-pad chord map (rl#358 proposal).

Renders the REAL GCR chord registry (net/src/controls.rs, 24 codes) as a sparse
4-ary tree in the Poincare disk. Each d-pad press applies a Mobius translation
that glides the pressed child to the disk center, so the exponential code space
always fits on screen and the motion itself IS the code you are typing.
Includes the rl#358 amendment (unlock growth: a sealed subtree teased as a bud,
then unfolding) and the rl#359 hook (each direction is a note; pitch glyphs on
edges, melody strip below).

Run: nix-shell -p python3Packages.pillow dejavu_fonts ffmpeg --run \
       'python3 render.py && ffmpeg -y -framerate 20 -i frames/f%04d.png \
        -vf "split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer" \
        dpad-map-hyperbolic.gif'
"""

import cmath
import math
import os

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- real data
# net/src/controls.rs GCR_CHORDS, verbatim codes and labels.
U, D, L, R = "UDLR"
REGISTRY = [
    ("UL", "Enter plane"),
    ("UR", "Enter ship"),
    ("DD", "Exit to foot"),
    ("UUU", "Render: mesh"),
    ("UUD", "Render: mesh+colliders"),
    ("UUL", "Render: colliders"),
    ("DUUU", "Ground: bloom"),
    ("DUUD", "Ground: bloom aurora"),
    ("DUUL", "Ground: bloom ember"),
    ("DUUR", "Ground: bloom frost"),
    ("DUDU", "Ground: bloom rose"),
    ("DUDD", "Ground: bloom filigree"),
    ("DLU", "Ground: shipped"),
    ("DLD", "Ground: patterned"),
    ("DLL", "Ground: wind-combed"),
    ("DLR", "Ground: cracked loam"),
    ("DRU", "Ground: wet nocturne"),
    ("DRD", "Ground: watershed"),
    ("DRL", "Ground: naturalist"),
    ("DRR", "Ground: nocturne"),
    ("L", "Swap brain"),
    ("R", "Restart round"),
    ("LR", "Perf overlay"),
    ("UUDDLR", "Quit round"),
]
# The unlock demo: the bloom family (DU*) starts sealed, teased as a bud at DU.
LOCKED_PREFIX = "DU"

SIZE = 640
SS = 2  # supersample
W = SIZE * SS
FPS = 20
DISK_R = 0.94 * W / 2
CX, CY = W / 2, W / 2 - 14 * SS

BG = (14, 17, 23)
DIR_COLOR = {U: (79, 195, 247), D: (255, 183, 77), L: (186, 104, 250), R: (129, 199, 132)}
DIR_ANGLE = {U: 90, R: 0, D: 270, L: 180}  # math degrees, y-up
PITCH = {L: 0, D: 1, R: 2, U: 3}  # rl#359: each direction is a note; L lowest.
NOTE_NAME = {L: "A", D: "C", R: "E", U: "G"}
LOCK_GRAY = (95, 100, 108)
TRAIL = (240, 235, 220)
TEXT = (232, 232, 232)

FONT = "/run/current-system/sw/share/X11/fonts/DejaVuSans.ttf"
for cand in [FONT, os.environ.get("DPAD_FONT", "")]:
    if cand and os.path.exists(cand):
        FONT = cand
        break
else:
    import subprocess

    FONT = subprocess.run(
        ["fc-match", "-f", "%{file}", "DejaVu Sans"], capture_output=True, text=True
    ).stdout.strip()


def font(px):
    return ImageFont.truetype(FONT, px * SS)


# ---------------------------------------------------------------- tree layout
class Node:
    def __init__(self, code):
        self.code = code
        self.label = None
        self.children = {}
        self.z = 0j  # Poincare position
        self.heading = 0.0  # unused at root


def build_tree():
    root = Node("")
    for code, label in REGISTRY:
        n = root
        for c in code:
            n = n.children.setdefault(c, Node(n.code + c))
        n.label = label
    return root


STEP = 0.62  # euclidean offset of a child when its parent sits at the origin


def mobius(a, z):
    """Translate: the isometry taking 0 -> a."""
    return (a + z) / (1 + a.conjugate() * z)


def layout(node, incoming=None):
    """Place children. Root children sit at compass angles (U up, R right...);
    deeper children keep the same compass rose but rotated into the node's
    outward frame, so 'press U' is always the same local gesture."""
    for d, child in node.children.items():
        if incoming is None:
            ang = math.radians(DIR_ANGLE[d])
        else:
            # Relative turn vs the direction we arrived by, compressed so the
            # whole subtree stays in the outward wedge (a reverse turn maps to
            # +-86deg, still outward): straight=0, side turns +-43, reverse +-86.
            rel = (DIR_ANGLE[d] - DIR_ANGLE[incoming] + 540) % 360 - 180
            ang = node.heading + math.radians(rel * 0.48)
        off = cmath.rect(STEP, ang)
        child.z = mobius(node.z, off)
        # Child heading: away from the parent, measured in the child's frame.
        back = (node.z - child.z) / (1 - child.z.conjugate() * node.z)
        child.heading = cmath.phase(back) + math.pi
        layout(child, d)


def walk(node):
    yield node
    for c in node.children.values():
        yield from walk(c)


# ---------------------------------------------------------------- drawing
def recenter(a, z):
    return (z - a) / (1 - a.conjugate() * z)


def to_px(z):
    return (CX + z.real * DISK_R, CY - z.imag * DISK_R)


def geodesic(dr, a, b, color, width):
    """Poincare geodesic as a sampled polyline (exact: straight in a's frame)."""
    w = (b - a) / (1 - a.conjugate() * b)
    pts = [to_px(mobius(a, t / 14 * w)) for t in range(15)]
    dr.line(pts, fill=color, width=max(1, int(width)), joint="curve")
    return pts


def ease(t):
    return t * t * (3 - 2 * t)


def geo_lerp(c, t):
    """Point at fraction t of the geodesic from 0 to c."""
    if abs(c) < 1e-9:
        return 0j
    return cmath.rect(math.tanh(t * math.atanh(abs(c))), cmath.phase(c))


class Scene:
    def __init__(self):
        self.root = build_tree()
        layout(self.root)
        self.nodes = list(walk(self.root))
        self.by_code = {n.code: n for n in self.nodes}
        self.center = 0j  # Mobius offset currently applied
        self.path = ""  # code typed so far
        self.melody = []  # (dir, age_frames)
        self.unlock_t = 0.0  # 0 sealed .. 1 fully grown
        self.press = None  # (dir, frames_left) input flash
        self.exec_flash = 0
        self.banner = None

    def locked(self, code):
        return code.startswith(LOCKED_PREFIX) and len(code) > len("D") and self.unlock_t < 1

    def draw(self, out_path, subtitle=""):
        img = Image.new("RGB", (W, W), BG)
        dr = ImageDraw.Draw(img, "RGBA")
        dr.ellipse(
            [CX - DISK_R, CY - DISK_R, CX + DISK_R, CY + DISK_R],
            outline=(60, 66, 78),
            width=2 * SS,
        )
        a = self.center
        pos = {n.code: recenter(a, n.z) for n in self.nodes}
        bud = None

        # Edges first.
        for n in self.nodes:
            for d, ch in n.children.items():
                if self.locked(ch.code):
                    if ch.code == LOCKED_PREFIX and self.unlock_t == 0:
                        bud = pos[ch.code]
                        # The teased branch: a dim edge leads to the sealed bud.
                        geodesic(dr, pos[n.code], bud, LOCK_GRAY + (140,), 2 * SS)
                    if self.unlock_t == 0:
                        continue
                pz, cz = pos[n.code], pos[ch.code]
                if self.locked_growing(ch.code):
                    cz = self.grow_pos(pos, ch.code)
                fade = 1 - abs(cz)
                on_trail = self.path.startswith(ch.code)
                col = TRAIL if on_trail else DIR_COLOR[d]
                alpha = int(255 * min(1, 0.4 + fade) * self.node_alpha(ch.code))
                wdt = (4.0 if on_trail else 2.6) * SS * max(0.4, fade)
                pts = geodesic(dr, pz, cz, col + (alpha,), wdt)
                # rl#359 pitch glyph: a 4-rung ladder at the edge midpoint, the
                # rung for this direction's note lit.
                if fade > 0.42 and self.node_alpha(ch.code) > 0.6:
                    mx, my = pts[7]
                    s = 3.2 * SS * fade
                    for rung in range(4):
                        y = my + (1.5 - rung) * 2.6 * s / 3
                        lit = rung == PITCH[d]
                        c2 = col if lit else (70, 76, 86)
                        r2 = s * (0.42 if lit else 0.2)
                        dr.ellipse([mx - r2, y - r2, mx + r2, y + r2], fill=c2 + (alpha,))

        # Sealed bud: the teased locked branch.
        if bud is not None:
            self.draw_bud(dr, pos)

        # Nodes.
        for n in self.nodes:
            if n.code == "":
                continue
            if self.locked(n.code) and self.unlock_t == 0:
                continue
            z = self.grow_pos(pos, n.code) if self.locked_growing(n.code) else pos[n.code]
            fade = 1 - abs(z)
            if fade < 0.03:
                continue
            x, y = to_px(z)
            d = n.code[-1]
            alpha = self.node_alpha(n.code)
            r = (11.0 if n.label else 5.5) * SS * max(0.3, fade)
            col = DIR_COLOR[d]
            if n.label:
                dr.ellipse(
                    [x - r, y - r, x + r, y + r], fill=col + (int(230 * alpha),)
                )
            else:
                dr.ellipse(
                    [x - r, y - r, x + r, y + r],
                    outline=col + (int(200 * alpha),),
                    width=max(1, int(1.8 * SS)),
                )
            if n.code == self.path:
                rr = r + 5 * SS + (2 * SS if self.exec_flash else 0)
                glow = (255, 255, 255, 230) if self.exec_flash else TRAIL + (200,)
                dr.ellipse([x - rr, y - rr, x + rr, y + rr], outline=glow, width=3 * SS)
            if n.label and fade > 0.3 and alpha > 0.55:
                f = font(int(9 + 8 * fade))
                dr.text(
                    (x + r + 3 * SS, y),
                    n.label,
                    font=f,
                    fill=TEXT + (int(255 * alpha * min(1, (fade - 0.24) * 3.2)),),
                    anchor="lm",
                )

        self.draw_hud(dr, subtitle)
        img.resize((SIZE, SIZE), Image.LANCZOS).save(out_path)

    def node_alpha(self, code):
        if code.startswith(LOCKED_PREFIX) and len(code) > 2:
            return self.unlock_t
        return 1.0

    def locked_growing(self, code):
        return code.startswith(LOCKED_PREFIX) and len(code) > 2 and 0 < self.unlock_t < 1

    def grow_pos(self, pos, code):
        t = ease(self.unlock_t)
        anchor = pos[LOCKED_PREFIX]
        return anchor + (pos[code] - anchor) * t

    def draw_bud(self, dr, pos):
        z = pos[LOCKED_PREFIX]
        x, y = to_px(z)
        fade = 1 - abs(z)
        if fade < 0.05:
            return
        r = 11 * SS * fade * (1 + 0.12 * math.sin(self.pulse))
        dr.ellipse(
            [x - r, y - r, x + r, y + r],
            outline=LOCK_GRAY + (220,),
            width=2 * SS,
        )
        # Dashed halo suggests sealed content without revealing it.
        for k in range(10):
            a0 = self.pulse * 0.5 + k * math.tau / 10
            p0 = (x + math.cos(a0) * r * 1.7, y + math.sin(a0) * r * 1.7)
            dr.ellipse([p0[0] - SS, p0[1] - SS, p0[0] + SS, p0[1] + SS], fill=LOCK_GRAY + (150,))
        if fade > 0.4:
            dr.text((x, y), "?", font=font(int(12 * fade + 4)), fill=LOCK_GRAY + (255,), anchor="mm")
            dr.text(
                (x, y + r + 9 * SS),
                "6 sealed",
                font=font(10),
                fill=LOCK_GRAY + (255,),
                anchor="mm",
            )

    def draw_hud(self, dr, subtitle):
        dr.text((16 * SS, 12 * SS), "CHORD MAP", font=font(15), fill=TEXT + (255,))
        code_txt = "".join({"U": "^", "D": "v", "L": "<", "R": ">"}[c] for c in self.path)
        dr.text(
            (16 * SS, 34 * SS),
            "code: " + (code_txt or "-"),
            font=font(12),
            fill=(170, 178, 190, 255),
        )
        if subtitle:
            dr.text((W / 2, W - 12 * SS), subtitle, font=font(12), fill=(150, 158, 170, 255), anchor="mm")
        if self.banner:
            dr.text((W / 2, 26 * SS), self.banner, font=font(16), fill=(255, 244, 214, 255), anchor="mm")

        # Melody strip (rl#359): the code so far as a pitch contour.
        sx, sy = 16 * SS, W - 64 * SS
        if self.melody:
            prev = None
            for i, (d, _) in enumerate(self.melody[-10:]):
                x = sx + 10 * SS + i * 22 * SS
                y = sy + (3 - PITCH[d]) * 9 * SS
                if prev:
                    dr.line([prev, (x, y)], fill=(120, 128, 140, 200), width=SS)
                r = 4.6 * SS
                dr.ellipse([x - r, y - r, x + r, y + r], fill=DIR_COLOR[d] + (255,))
                dr.text((x, sy + 34 * SS), NOTE_NAME[d], font=font(9), fill=DIR_COLOR[d] + (255,), anchor="mm")
                prev = (x, y)

        # D-pad widget bottom-right, pressed direction lit.
        px, py, s = W - 58 * SS, W - 58 * SS, 15 * SS
        for d, (ox, oy) in {U: (0, -1), D: (0, 1), L: (-1, 0), R: (1, 0)}.items():
            x, y = px + ox * s * 1.25, py + oy * s * 1.25
            lit = self.press and self.press[0] == d
            col = DIR_COLOR[d] if lit else (58, 64, 74)
            dr.rounded_rectangle([x - s / 2, y - s / 2, x + s / 2, y + s / 2], 3 * SS, fill=col)
        if self.press:
            dr.text(
                (px, py + s * 2.6),
                f"{NOTE_NAME[self.press[0]]}4",
                font=font(11),
                fill=DIR_COLOR[self.press[0]] + (255,),
                anchor="mm",
            )

    pulse = 0.0


# ---------------------------------------------------------------- storyboard
def main():
    os.makedirs("frames", exist_ok=True)
    sc = Scene()
    frames = []

    def hold(n, subtitle=""):
        for _ in range(n):
            frames.append(("hold", subtitle))

    def press(d, n_anim, subtitle=""):
        frames.append(("press", d))
        for i in range(n_anim):
            frames.append(("anim", (d, (i + 1) / n_anim, subtitle)))

    def unlock(n):
        for i in range(n):
            frames.append(("unlock", (i + 1) / n))

    def zoom_home(n):
        for i in range(n):
            frames.append(("home", (i + 1) / n))

    hold(28, "hold X, tap the d-pad: every command is a place")
    press(D, 22, "v  glide: the pressed branch becomes the center")
    hold(10, "")
    press(U, 22, "^  a sealed branch: content teased, not shown")
    hold(20, "unlock: Ground styles")
    unlock(36)
    hold(12, "the map GROWS: 6 new commands bloom in place")
    press(U, 20, "^  deeper: the rim always has room")
    hold(8, "")
    press(D, 18, "v  Ground: bloom aurora")
    frames.append(("exec", None))
    hold(18, "release X: the code at center executes")
    zoom_home(30)
    hold(26, "every path is a melody (rl#359): pitch glyphs on each edge")

    start = 0j
    seg_from = 0j
    cur_node = sc.by_code[""]
    for i, (kind, arg) in enumerate(frames):
        sc.pulse += 0.35
        sc.exec_flash = max(0, sc.exec_flash - 1)
        subtitle = ""
        if kind == "press":
            d = arg
            sc.press = (d, 6)
            sc.melody.append((d, 0))
            sc.path += d
            seg_from = sc.center
            cur_node = sc.by_code[sc.path]
        elif kind == "anim":
            d, t, subtitle = arg
            # Interpolate the Mobius offset along the geodesic to the target.
            sc.center = _blend(seg_from, cur_node.z, ease(t))
        elif kind == "unlock":
            sc.unlock_t = arg
            sc.banner = "UNLOCKED: Ground styles" if arg < 0.999 else None
        elif kind == "exec":
            sc.exec_flash = 14
            sc.banner = "Ground: bloom aurora"
        elif kind == "home":
            sc.banner = None
            sc.center = _blend(sc._home_from, 0j, ease(arg))
        elif kind == "hold":
            subtitle = arg
        if kind != "home":
            sc._home_from = sc.center
        if sc.press:
            sc.press = None if sc.press[1] <= 1 else (sc.press[0], sc.press[1] - 1)
        sc.draw(f"frames/f{i:04d}.png", subtitle)
        if i % 40 == 0:
            print(f"frame {i}/{len(frames)}")
    print(f"{len(frames)} frames")


def _blend(a, b, t):
    """Geodesic blend of two Mobius offsets: move a toward b."""
    # Translate so a -> 0, find b in that frame, walk fraction t, map back.
    w = (b - a) / (1 - a.conjugate() * b)
    return mobius(a, geo_lerp(w, t))


if __name__ == "__main__":
    main()
