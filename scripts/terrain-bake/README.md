# terrain-bake — offline heightmap baking (rl#281)

Bakes seeded heightmap artifacts from
[xandergos/terrain-diffusion](https://github.com/xandergos/terrain-diffusion)
(SIGGRAPH '26 InfiniteDiffusion, MIT). Runtime never runs the model — it reads
the committed artifact under `crab-world/assets/terrain/`.

Two pieces, split on the trust boundary:

- `serve-model.sh` — clones + installs + runs the upstream model API inside
  `run-untrusted -g` (no `$HOME`, GPU passed through). Downloads HF weights and
  WorldClim rasters into its work dir on first run (~1.5G).
- `bake.py` — trusted client, runs outside the sandbox. Fetches an elevation
  tile (the API returns int16 meters), optionally adds a seeded FBM detail
  octave, writes the `.terrain` artifact + a shaded-relief preview PNG.

## Bake a world

```bash
scripts/terrain-bake/serve-model.sh &   # wait for "Running on"
nix-shell -p 'python312.withPackages (ps: [ps.numpy ps.pillow])' \
  --run 'python3 scripts/terrain-bake/bake.py --seed 281 --size 1024 \
    --origin 1536 -2560 \
    --out crab-world/assets/terrain/gcr-seed281.terrain \
    --preview crab-world/assets/terrain/gcr-seed281-preview.png'
```

The world is infinite and deterministic per seed; `--origin`/`--size` pick a
window (probe small tiles to scout for good land — `/terrain?...&seed=N`
returns stats-worthy int16 directly). `gcr-seed281` came from origin
(1536, -2560), a 1024² all-land mountain block, elevation 295–4508 m.

## Artifact format (`.terrain`)

Little-endian: `b"RLTERR01"` · `u32` JSON length · JSON metadata ·
`rows*cols` int16 elevation meters, row-major.

Orientation: `grid[row][col]`, `origin` is the window's top-left `(i, j)` in
the model's target-resolution pixel space (i = row, j = col), and row 0 is the
top of the preview PNG. The world-axis mapping (which of row/col is x vs z) is
the runtime sampler's decision, not the artifact's — the artifact only promises
it matches its own preview.

Metadata carries the scale knobs the runtime sampler applies — `cell_size_m`
(horizontal stretch) and `height_scale` (vertical exaggeration) — plus full
provenance (seed, origin, model + `model_rev`, optional FBM params). Knobs are
declarative: re-tuning world scale edits metadata, no re-bake, no GPU.
`serve-model.sh` pins the upstream rev (`UPSTREAM_REV`) so a bake is
reproducible; bump the pin deliberately, alongside a re-bake.
