# rl#398 — MP: joined peer's command map renders empty

Two-peer repro (offscreen lavapipe, real iroh, `game net-screenshot` ×2 on one
box): host runs with a seeded discovered-code save (`UL UR DD UUU UUD` via
`--chord-map-file`), joined client runs with no save; both hold the chord
modifier from frame 120 (`--chord-hold-at 120`, never released) so the shot
captures the open combo map at frame 200.

- `before-host.png` / `before-client.png` (build 36cc5757 + the repro flags):
  the host's map is populated; the joined client's map is a bare root ring —
  the client LACKS THE DATA (discovery is a per-device save, nothing
  replicates), not a render failure. The same run with the seeded peer forming
  as the joined client showed a populated client map, confirming the render
  path was never role-sensitive.
- `after-host.png` / `after-client.png` (fix landed): same setup; the joined
  client now renders the host's full map — the host stamps its discovered set
  onto every outgoing `CoreSnapshot` (rider metadata like `input_next`), and
  every peer renders local save ∪ replicated session set. Replicated codes are
  session-only: never persisted, never relayed onward.

Repro:

```
game net-screenshot --host --settle 200 --chord-hold-at 120 \
    --chord-map-file host-map.txt --out host.png &
game net-screenshot --join --settle 200 --chord-hold-at 120 --out client.png
```

(`--host`/`--join` don't pin which peer forms as the server — read the
`formed as host|client` line and label the shots by role.)
