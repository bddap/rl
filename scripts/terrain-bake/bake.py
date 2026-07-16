#!/usr/bin/env python3
"""Bake a terrain heightmap artifact from a running terrain-diffusion API.

Fetches a seeded elevation tile from the model server (serve-model.sh),
optionally adds a seeded FBM detail octave (the model output is band-limited;
detail below its native pitch has to come from noise), and writes:

  - the .terrain artifact: magic + JSON metadata + raw int16-le grid
  - a shaded-relief preview PNG

The scale knobs (cell_size_m horizontal, height_scale vertical) are DECLARED in
the metadata, not baked into the samples: the runtime sampler applies them, so
re-tuning world scale never requires a re-bake or GPU. FBM does mutate samples
(it adds information), so its parameters are recorded for provenance.

Artifact layout, little-endian throughout:
  b"RLTERR01" | u32 json_len | json_len bytes of JSON metadata | w*h int16 rows

Runs outside the sandbox (trusted, first-party): stdlib + numpy + Pillow.
  nix-shell -p 'python312.withPackages (ps: [ps.numpy ps.pillow])' \
    --run 'python3 scripts/terrain-bake/bake.py --seed 281 ...'
"""

import argparse
import json
import struct
import urllib.request

import numpy as np

MAGIC = b"RLTERR01"


def fetch_elevation(api: str, seed: int, origin: tuple[int, int], size: int, scale: int) -> np.ndarray:
    i1, j1 = origin
    url = (
        f"{api}/terrain?i1={i1}&j1={j1}&i2={i1 + size}&j2={j1 + size}"
        f"&scale={scale}&seed={seed}"
    )
    with urllib.request.urlopen(url, timeout=3600) as resp:
        h = int(resp.headers["X-Height"])
        w = int(resp.headers["X-Width"])
        data = resp.read()
    elev = np.frombuffer(data[: h * w * 2], dtype="<i2").reshape(h, w)
    assert (h, w) == (size, size), f"server returned {h}x{w}, wanted {size}x{size}"
    return elev


def fbm(shape: tuple[int, int], octaves: int, wavelength_px: float, seed: int) -> np.ndarray:
    """Value-noise FBM in [-1, 1], persistence 0.5, lacunarity 2."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    amp, wl, total = 1.0, wavelength_px, 0.0
    for _ in range(octaves):
        cells = (max(2, int(np.ceil(shape[0] / wl)) + 1), max(2, int(np.ceil(shape[1] / wl)) + 1))
        lattice = rng.uniform(-1.0, 1.0, cells).astype(np.float32)
        yi = np.linspace(0, cells[0] - 1.001, shape[0], dtype=np.float32)
        xi = np.linspace(0, cells[1] - 1.001, shape[1], dtype=np.float32)
        y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
        ty, tx = yi - y0, xi - x0
        # smoothstep for C1 continuity at lattice lines
        ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
        ty, tx = ty[:, None], tx[None, :]
        v = (
            lattice[y0, :][:, x0] * (1 - ty) * (1 - tx)
            + lattice[y0 + 1, :][:, x0] * ty * (1 - tx)
            + lattice[y0, :][:, x0 + 1] * (1 - ty) * tx
            + lattice[y0 + 1, :][:, x0 + 1] * ty * tx
        )
        out += amp * v
        total += amp
        amp *= 0.5
        wl /= 2.0
    return out / total


def hillshade_preview(elev: np.ndarray, cell_size_m: float, path: str) -> None:
    from PIL import Image

    e = elev.astype(np.float32)
    gy, gx = np.gradient(e, cell_size_m)
    # Lambertian shade, light from NW at 45 degrees elevation.
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)

    # Hypsometric tint: sea in blues, land green -> brown -> white.
    sea = e <= 0
    land_t = np.zeros_like(e)
    if (~sea).any():
        lmax = max(float(e[~sea].max()), 1.0)
        land_t = np.clip(e / lmax, 0, 1)
    sea_t = np.zeros_like(e)
    if sea.any():
        smin = min(float(e[sea].min()), -1.0)
        sea_t = np.clip(e / smin, 0, 1)

    stops = np.array(
        [[70, 120, 50], [110, 140, 60], [170, 140, 80], [150, 120, 100], [235, 235, 235]],
        dtype=np.float32,
    )
    idx = land_t * (len(stops) - 1)
    lo = np.clip(idx.astype(int), 0, len(stops) - 2)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    deep, shallow = np.array([15, 30, 80], np.float32), np.array([60, 110, 180], np.float32)
    rgb[sea] = shallow * (1 - sea_t[sea, None]) + deep * sea_t[sea, None]

    lit = rgb * (0.35 + 0.65 * shade[..., None])
    Image.fromarray(lit.clip(0, 255).astype(np.uint8)).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://127.0.0.1:8017")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--size", type=int, default=1024, help="grid edge, pixels")
    p.add_argument("--origin", type=int, nargs=2, default=[0, 0], metavar=("I", "J"),
                   help="window top-left (i=row, j=col) in target-resolution pixels")
    p.add_argument("--api-scale", type=int, default=1, help="upsample factor relative to model-native pitch")
    p.add_argument("--model-rev", default=None, help="upstream terrain-diffusion git rev (provenance)")
    p.add_argument("--native-pitch-m", type=float, default=30.0, help="model-native meters/pixel")
    p.add_argument("--cell-size-m", type=float, default=None,
                   help="declared horizontal stretch knob (default: native pitch / api-scale)")
    p.add_argument("--height-scale", type=float, default=1.0, help="declared vertical exaggeration knob")
    p.add_argument("--fbm-octaves", type=int, default=0, help="0 disables the detail octave")
    p.add_argument("--fbm-amplitude-m", type=float, default=8.0)
    p.add_argument("--fbm-wavelength-px", type=float, default=64.0)
    p.add_argument("--out", required=True)
    p.add_argument("--preview", required=True)
    args = p.parse_args()

    elev = fetch_elevation(args.api, args.seed, tuple(args.origin), args.size, args.api_scale)

    fbm_meta = None
    if args.fbm_octaves > 0:
        detail = fbm(elev.shape, args.fbm_octaves, args.fbm_wavelength_px, args.seed)
        elev = np.clip(
            elev.astype(np.float32) + detail * args.fbm_amplitude_m, -32768, 32767
        ).astype("<i2")
        fbm_meta = {
            "octaves": args.fbm_octaves,
            "amplitude_m": args.fbm_amplitude_m,
            "wavelength_px": args.fbm_wavelength_px,
            "seed": args.seed,
        }

    cell = args.cell_size_m if args.cell_size_m is not None else args.native_pitch_m / args.api_scale
    meta = {
        "rows": elev.shape[0],
        "cols": elev.shape[1],
        # Fixed by construction (numpy row-major + PIL): stage-2 readers rely
        # on this string matching reality, not on guessing.
        "layout": "row-major int16-le; grid[row][col]; origin=(i,j)=top-left; row 0 = preview PNG top",
        "cell_size_m": cell,
        "height_scale": args.height_scale,
        "seed": args.seed,
        "origin": list(args.origin),
        "api_scale": args.api_scale,
        "model": "xandergos/terrain-diffusion-30m",
        "model_rev": args.model_rev,
        "fbm": fbm_meta,
        "elev_min_m": int(elev.min()),
        "elev_max_m": int(elev.max()),
    }

    blob = json.dumps(meta).encode()
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
        f.write(np.ascontiguousarray(elev, dtype="<i2").tobytes())

    hillshade_preview(elev, cell, args.preview)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
