//! `rl` library surface.
//!
//! Currently exposes only the multiplayer netcode foundation ([`net`]) so the
//! `game` binary and the headless determinism tests can share it. The training app
//! (`src/main.rs`) is a separate binary that does not depend on this library — the
//! two coexist in one crate during the game's bring-up.

pub mod net;
