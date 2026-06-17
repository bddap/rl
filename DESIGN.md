# rl — Crab Combat: Train Your Bot

## Overview

A game where players train AI-controlled crab bots for combat through adversarial play.
The player fights their own bot in a 3D third-person arena. The bot is controlled by an
end-to-end neural network whose weights are updated via reinforcement learning after each
session. Bot bodies are physically simulated — every limb, claw, and joint is driven by
motor actuators commanded by the network. Over time, the bot learns to walk, fight, and
adapt to its owner's playstyle.

Long-term: bots can lose limbs to physics-based damage and must learn to compensate.

## Core Loop

```
 ┌─────────────────────────────────────────────────┐
 │                  GAME SESSION                    │
 │                                                  │
 │   Player (3rd person) ──fights──▶ Crab Bot      │
 │                                    │             │
 │                              NN controls         │
 │                              joint motors        │
 │                                    │             │
 │                              Rapier physics      │
 │                              steps the world     │
 │                                    │             │
 │                              Observations +      │
 │                              rewards collected    │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  RL Training  │
              │  (burn)       │
              │               │
              │  Update NN    │
              │  weights      │
              └───────┬───────┘
                      │
                      ▼
               Next session:
               bot is smarter
```

## Architecture

### Crate Structure

Single crate to start. Split into modules, extract crates later if needed.

```
src/
  main.rs              — entry point, Bevy app setup
  bot/
    mod.rs             — bot module root
    body.rs            — crab body definition (limbs, joints, colliders)
    brain.rs           — neural network definition (burn)
    actuator.rs        — maps NN outputs to joint motor commands
    sensor.rs          — builds observation vector from physics state
  physics/
    mod.rs             — physics module root
    world.rs           — arena, gravity, physics pipeline config
    damage.rs          — physics-based damage (impact forces → HP)
  player/
    mod.rs             — player module root
    controller.rs      — 3rd person player input → character control
    camera.rs          — 3rd person camera rig
  training/
    mod.rs             — training module root
    session.rs         — episode management, reward calculation
    algorithm.rs       — RL algorithm (algorithm-agnostic trait)
    replay.rs          — experience replay buffer
  combat/
    mod.rs             — combat module root
    arena.rs           — arena geometry, spawn points
    scoring.rs         — damage tracking, round outcomes
```

### Tech Stack

| Concern              | Choice              | Rationale                                                |
|----------------------|---------------------|----------------------------------------------------------|
| Language             | Rust 2024           | Performance, safety, single language for sim + ML        |
| Game engine          | Bevy                | ECS, rendering, input, asset pipeline, bevy_rapier       |
| Physics              | Rapier 3D           | Articulated multibodies, joint motors, determinism, fast |
| Neural networks      | burn                | Pure Rust, autodiff, GPU backends, transformers built-in |
| RL algorithms        | Custom on burn      | No existing burn-rl crate; build PPO/SAC on primitives   |

### Dependencies (Cargo.toml)

```toml
[dependencies]
bevy = "0.15"
bevy_rapier3d = "0.29"
burn = { version = "0.20", features = ["autodiff", "ndarray", "wgpu"] }
rand = "0.8"
serde = { version = "1", features = ["derive"] }
```

## Crab Bot Body

### Why Crabs

Carcinization is convergent evolution's answer to "what body plan works?" Low center of
gravity, wide stable base, multiple legs for redundancy, claws for manipulation and
combat, armored carapace. Also: losing a leg is survivable, which makes the limb-damage
goal tractable.

### Body Plan

Modeled as a Rapier `MultibodyJointSet` tree:

```
                    [Carapace]  (root, main body)
                   /    |     \
           [Eye_L] [Eye_R]  [Mouth]
          /                        \
   [Claw_L]                    [Claw_R]
   upper arm ─ revolute          upper arm ─ revolute
   forearm   ─ revolute          forearm   ─ revolute
   pincer    ─ revolute         pincer    ─ revolute
        \                            /
     [Leg_L1..L4]            [Leg_R1..R4]
     each: coxa  ─ revolute (yaw)
           femur ─ revolute (pitch)
           tibia ─ revolute (pitch)
```

**Joint count:** ~35 actuated DOF
- 8 legs × 3 joints = 24
- 2 claws × 3 joints = 6
- 2 eyes × 1 joint = 2
- Mouth × 1 = 1
- Carapace orientation relative to leg base = 2 (pitch/roll)

Every joint is a Rapier `RevoluteJoint` (the pincers included). The NN outputs a
target torque per joint each timestep, applied directly to the joint — there is no
velocity or position servo.

### Physical Properties

- Carapace: heavy, large cuboid/convex hull, high friction
- Legs: light, capsule-shaped segments, medium friction on tips
- Claws: medium weight, box-shaped with pincer collider
- Total mass: tuned so the bot can stand and walk under default gravity
- All segments have colliders for combat contact detection

## Neural Network

### Architecture

End-to-end: observations in, joint commands out.

```
Observations (state vector)
       │
       ▼
  ┌─────────┐
  │  Input   │  Linear embedding, LayerNorm
  │  Encoder │
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │Transformer│  Small (2-4 layers, 4-8 heads)
  │  Encoder  │  Processes relationships between body parts
  └────┬──────┘  (each limb's state as a "token")
       │
       ▼
  ┌─────────┐
  │  Action  │  Linear layers → joint targets
  │  Head    │  tanh activation (bounded output)
  └────┬─────┘
       │
       ▼
  Joint motor commands (continuous, one per DOF)
```

The transformer is motivated by the structured, multi-part nature of the body. Each limb
group's proprioceptive state is a natural "token." Attention lets the network learn
inter-limb coordination (e.g., shifting weight to compensate for a missing leg).

For the RL value function, a parallel head:

```
       ... (shared transformer trunk)
       │
       ▼
  ┌─────────┐
  │  Value   │  Linear layers → scalar value estimate
  │  Head    │
  └─────────┘
```

### Observation Space (State Vector)

Per-limb proprioception (for each of ~35 joints):
- Joint angle (1)
- Joint angular velocity (1)
- Motor torque applied last step (1)

Body state:
- Carapace position (3) — relative to arena center
- Carapace orientation (4) — quaternion
- Carapace linear velocity (3)
- Carapace angular velocity (3)

Enemy state (relative to bot):
- Relative position (3)
- Relative velocity (3)
- Relative orientation (4)

Combat state:
- Own HP (1)
- Enemy HP (1)
- Contact forces on each body segment (per-segment scalar, ~20)

**Total observation dimension:** ~35×3 + 13 + 10 + 22 ≈ **150 floats**

### Action Space

Continuous, one float per actuated DOF:
- ~35 joint target velocities, each in [-1, 1], scaled to joint-specific max velocity

**Total action dimension:** ~**35 floats**

## RL Training

### Session Flow

1. **Reset:** Spawn player and bot in arena at starting positions
2. **Play:** Player fights bot in real-time. Each physics step:
   - Build observation vector from physics state
   - Forward pass through NN → joint commands
   - Apply commands to Rapier joint motors
   - Step physics
   - Compute step reward
   - Store transition (obs, action, reward, next_obs, done) in buffer
3. **Episode end:** Timer expires or one combatant reaches 0 HP
4. **Train:** Run RL update on collected transitions
5. **Loop:** Start next session with updated weights

### Reward Function

Shaped reward to bootstrap learning. Curriculum:

**Phase 1 — Stand up:**
- `+1.0` per step if carapace is above minimum height
- `-1.0` per step if carapace touches ground
- Small bonus for low energy expenditure (encourages efficiency)

**Phase 2 — Locomotion:**
- Phase 1 rewards, plus:
- `+0.5` for moving toward a target waypoint
- `-0.1` per step (time pressure)

**Phase 3 — Combat:**
- `+10.0` for dealing damage to player (impact force above threshold)
- `-5.0` for taking damage
- `+100.0` for winning the round
- `-50.0` for losing
- `+0.1` for facing the player (encourages engagement)
- `-0.01` per step (slight time pressure)

Phase transitions are manual (config flag) or automatic (trigger on reward threshold).

### Algorithm Interface

```rust
trait RlAlgorithm {
    fn select_action(&self, observation: &Tensor<B, 1>) -> Tensor<B, 1>;
    fn store_transition(&mut self, transition: Transition);
    fn update(&mut self) -> TrainingMetrics;
    fn save(&self, path: &Path);
    fn load(&mut self, path: &Path);
}
```

First implementation: PPO (well-understood for continuous control, on-policy so the
real-time play sessions map naturally to episode collection).

Stretch: SAC for better sample efficiency once the infrastructure is proven.

### Replay Buffer

- PPO: rollout buffer, discarded after each update (on-policy)
- SAC: circular replay buffer with uniform sampling (off-policy)
- Transitions stored as: `(observation, action, reward, next_observation, done, log_prob)`

### Training Acceleration

- **Headless mode:** Skip rendering, run physics at max speed for batch training
- **Parallel arenas:** Multiple physics worlds stepping in parallel (rayon)
- **GPU inference:** Use burn's wgpu backend for batched NN forward passes
- **Mixed play/train:** Human sessions generate high-quality data; headless self-play
  generates volume

## Player

### Controls (3rd Person)

- WASD: movement
- Mouse: aim / camera orbit
- LMB: attack (melee or shoot depending on equipped weapon)
- RMB: block / alt-fire
- Space: jump / dodge
- Player character: simplified humanoid or another crab (TBD)

### Camera

- Third-person orbit camera, offset behind and above player
- Smooth follow with configurable distance and angle
- Free-look with right mouse drag

## Combat

### Physics-Based Damage

No hitpoint guns. Damage is computed from physics:
- **Impact force:** When a collider collision occurs, read the contact force magnitude
  from Rapier's contact events. Force above a threshold deals damage.
- **Damage = f(force, material, body_part):** Claws deal more damage than legs.
  Carapace absorbs more than joints.
- **Knockback:** Directly emergent from physics — no fake knockback impulses needed.

### Weapons (Later)

- Projectiles: small rigid bodies fired at high velocity. Damage on impact via same
  force-based system.
- Mounted guns: attached to claw or carapace joint, aimed by the NN.

### Limb Destruction (Stretch Goal)

- Each body segment has HP based on its mass and material
- When segment HP reaches 0, detach it from the multibody tree
  (remove the joint, spawn the segment as a free rigid body)
- The NN must adapt to the new body configuration
- The transformer architecture handles this naturally — missing limb tokens are
  zeroed or masked out, attention redistributes

## Arena

### v0.1 Arena

- Flat rectangular platform with walls
- Gravity: standard Earth-like
- Ground: high-friction surface
- Size: ~20m × 20m (enough room to maneuver, small enough for engagement)

### Later

- Obstacles, ramps, pits
- Multiple arena layouts
- Environmental hazards

## Milestones

### M0: Scaffold (current)
- [x] Rust project with nix shell
- [ ] Bevy app window opens
- [ ] Rapier physics world with ground plane

### M1: Crab Stands
- [ ] Crab body spawned as Rapier multibody
- [ ] Joint motors respond to fixed commands
- [ ] NN wired up (burn) — random outputs move joints
- [ ] Observation vector built from physics state
- [ ] Crab learns to stand via RL (phase 1 reward)

### M2: Crab Walks
- [ ] Phase 2 locomotion reward
- [ ] Crab learns basic locomotion toward a target
- [ ] Camera and basic rendering of crab body (debug meshes)

### M3: Player Enters
- [ ] 3rd person player controller
- [ ] Player can move around arena
- [ ] Player can hit the crab (melee)

### M4: Combat Training
- [ ] Phase 3 combat reward
- [ ] Bot fights back (learns to approach and attack player)
- [ ] Damage system (force-based)
- [ ] Round system (HP, win/lose, reset)
- [ ] Training persists across sessions (save/load weights)

### M5: Polish & Stretch
- [ ] Headless training mode (accelerated)
- [ ] Limb destruction
- [ ] Ranged weapons
- [ ] Multiple arena layouts
- [ ] Bot-vs-bot mode

## Open Questions

- **Transformer sizing:** How small can the transformer be and still learn inter-limb
  coordination? Start with 2 layers, 4 heads, 64-dim embeddings and tune.
- **Sim-to-play gap:** Physics runs at fixed timestep (e.g., 60Hz). NN inference must
  keep up in real-time play. Benchmark burn inference latency early.
- **Curriculum automation:** Can we auto-detect when to advance reward phases based on
  average episode return?
- **Body plan variation:** Later, can players customize their crab (more legs, bigger
  claws, asymmetric builds)? The transformer observation scheme supports variable
  token counts naturally.
- **Observation space for lost limbs:** Zero-masking vs. removing tokens vs. learned
  [MISSING] token embedding.
