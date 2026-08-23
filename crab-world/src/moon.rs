//! The moon — THE light source (rl#374). One knob-driven state drives both the
//! sky's visible disc (`sky.wgsl`, via `StarSkyMaterial` uniforms) and the one
//! `DirectionalLight`, so the light on the terrain and the disc in the sky can
//! never disagree. There is no other directional light anywhere.

use bevy::prelude::*;

/// Full-moon illuminance (lux, stylized — real moonlight is ~0.3 lux). Tuned so a
/// full-moon ground vista renders at 30% of the pre-moon static light's mean
/// linear luminance (rl#404 — measured on screenshots, not scaled on the
/// constant: the 400-lux ambient floor makes as-rendered ≠ proportional-in-lux).
const FULL_MOON_LUX: f32 = 2300.0;

/// The hue knob turns only the hue of an otherwise fixed pastel: saturation 1.0 /
/// lightness 0.925 in HSL, chosen so the default hue reproduces the pre-moon
/// light color exactly (`hsl(220°) == srgb(0.85, 0.90, 1.0)` — guarded by test).
const MOON_SATURATION: f32 = 1.0;
const MOON_LIGHTNESS: f32 = 0.925;

/// Sky motion (rl#374 motion stage) — sim seconds per real second. 24 means a
/// day-length traversal passes in one real hour and the ~29.5-day synodic phase
/// cycle in ~29.5 real hours (the epic's "1/24 timescale").
pub const DEFAULT_TIMESCALE: f32 = 24.0;
const DAY_SECONDS: f64 = 86_400.0;
const SYNODIC_SECONDS: f64 = 29.530588 * DAY_SECONDS;

/// The traversal path's anchor is the default (pre-moon-light) pose: the moon
/// "rises" there, so a freshly booted moving sky starts exactly where the
/// static stage-1 moon stood and departs smoothly.
const RISE_AZIMUTH_DEG: f32 = 43.8;
/// The elevation floor: the traversal never dips below this, so the moon never
/// sets and never grazes low enough for the missing far-field terrain shadows
/// to read as wrong. It equals the default elevation — the shipped stage-1
/// look already proved this height sound.
const ELEVATION_FLOOR_DEG: f32 = 21.5;
const PEAK_ELEVATION_DEG: f32 = 60.0;
/// How far the azimuth travels in one crossing before turning back — half the
/// sky, like a real moon's rise-to-set sweep.
const TRAVERSAL_SWEEP_DEG: f32 = 180.0;

/// The moon's runtime-tweakable knobs. Mutate the resource and everything moon
/// follows on the next frame: the disc, the light direction, the tint, the
/// luminosity. Angles are degrees (what you'd type on a flag).
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct Moon {
    /// Compass direction of the moon, degrees around +Y (0 = +Z, 90 = +X).
    pub azimuth_deg: f32,
    /// Height above the horizon, degrees (90 = zenith). Kept high in practice:
    /// terrain casts no far-field shadows, and a grazing moon would make that
    /// obvious (the epic's elevation floor — enforced by the motion stage).
    pub elevation_deg: f32,
    /// Hue of the moonlight AND the disc, degrees on the HSL wheel.
    pub hue_deg: f32,
    /// Synodic phase, wrapping [0, 1): 0 = new, 0.5 = full. Drives luminosity
    /// and the disc's terminator.
    pub phase: f32,
    /// Sky-motion speed, sim seconds per real second (rl#374): the traversal
    /// and the phase sweep both run this much faster than reality. 0 freezes
    /// the sky, which is what makes the azimuth/elevation knobs holdable —
    /// while moving, motion owns them. Phase stays a live offset either way
    /// (motion advances it incrementally).
    pub timescale: f32,
}

impl Default for Moon {
    /// Reproduces the pre-moon static light's pose and color: its euler-angle
    /// transform pointed the light along -(0.644, 0.367, 0.671) and its color
    /// was `hsl(220°)`. Illuminance is dimmer than that light was (rl#404).
    fn default() -> Self {
        Self {
            azimuth_deg: RISE_AZIMUTH_DEG,
            elevation_deg: ELEVATION_FLOOR_DEG,
            hue_deg: 220.0,
            phase: 0.5,
            timescale: DEFAULT_TIMESCALE,
        }
    }
}

impl Moon {
    /// Unit vector from the world origin toward the moon.
    pub fn direction(&self) -> Vec3 {
        let az = self.azimuth_deg.to_radians();
        let el = self.elevation_deg.to_radians();
        Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos())
    }

    /// Moonlight color — the one source for the light tint and the disc tint.
    pub fn color(&self) -> Color {
        Color::hsl(
            self.hue_deg.rem_euclid(360.0),
            MOON_SATURATION,
            MOON_LIGHTNESS,
        )
    }

    /// Sunlit fraction of the visible disc: 0 at new, 1 at full.
    pub fn illuminated_fraction(&self) -> f32 {
        (1.0 - (std::f32::consts::TAU * self.phase).cos()) / 2.0
    }

    /// Phase-driven luminosity. Cubic in the illuminated fraction — a quarter
    /// moon is ~1/10 of full, matching the real moon's strongly superlinear
    /// brightness curve — over a small earthshine floor so a new-moon world is
    /// dark but not a void (ambient starlight stays separate). Distinct from
    /// the disc's 0.1 dark-limb floor in `sky.wgsl` — that one is albedo on the
    /// visible disc, this one is light on the world; they need not match.
    pub fn illuminance(&self) -> f32 {
        let f = self.illuminated_fraction();
        FULL_MOON_LUX * (0.02 + 0.98 * f.powi(3))
    }
}

/// Gallop tempo ([`MoonSong::TempoGallop`]): a day-length traversal per real
/// minute, the synodic phase cycle in ~29.5 real minutes — fast enough to watch
/// the sky move while standing still.
const GALLOP_TIMESCALE: f32 = 1440.0;

/// What one d-pad song does to the moon (rl#374): each song poses one knob
/// family. The codes live in the surface's chord registry (the songs ARE chord
/// entries, so the instrument and the discovered-codes map carry them for
/// free); this enum is the one source for each song's effect, so a surface
/// maps action → song and never re-states moon semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonSong {
    /// Phase 0.5 — full light.
    PhaseFull,
    /// Phase 0.25 — waxing quarter.
    PhaseWaxing,
    /// Phase 0.75 — waning quarter.
    PhaseWaning,
    /// Phase 0.0 — near-dark.
    PhaseNew,
    /// The default 220° pastel — the shipped look.
    HueSilver,
    /// 10° — blood moon.
    HueBlood,
    /// 45° — harvest gold.
    HueHarvest,
    /// 140° — green.
    HueVerdant,
    /// Timescale back to [`DEFAULT_TIMESCALE`].
    TempoDrift,
    /// Timescale 0 — the sky holds still (the manual-pose escape hatch).
    TempoFreeze,
    /// Timescale [`GALLOP_TIMESCALE`] — watch the sky move.
    TempoGallop,
    /// Park the moon at the traversal's high point. Freezes the sky: while
    /// moving, motion owns the angles and the pose would be overwritten next
    /// frame.
    PoseZenith,
    /// Park the moon at its rise pose (the default). Freezes likewise.
    PoseRise,
}

impl MoonSong {
    pub const ALL: [Self; 13] = [
        Self::PhaseFull,
        Self::PhaseWaxing,
        Self::PhaseWaning,
        Self::PhaseNew,
        Self::HueSilver,
        Self::HueBlood,
        Self::HueHarvest,
        Self::HueVerdant,
        Self::TempoDrift,
        Self::TempoFreeze,
        Self::TempoGallop,
        Self::PoseZenith,
        Self::PoseRise,
    ];

    pub fn apply(self, moon: &mut Moon) {
        match self {
            Self::PhaseFull => moon.phase = 0.5,
            Self::PhaseWaxing => moon.phase = 0.25,
            Self::PhaseWaning => moon.phase = 0.75,
            Self::PhaseNew => moon.phase = 0.0,
            Self::HueSilver => moon.hue_deg = 220.0,
            Self::HueBlood => moon.hue_deg = 10.0,
            Self::HueHarvest => moon.hue_deg = 45.0,
            Self::HueVerdant => moon.hue_deg = 140.0,
            Self::TempoDrift => moon.timescale = DEFAULT_TIMESCALE,
            Self::TempoFreeze => moon.timescale = 0.0,
            Self::TempoGallop => moon.timescale = GALLOP_TIMESCALE,
            Self::PoseZenith => {
                moon.azimuth_deg = RISE_AZIMUTH_DEG + TRAVERSAL_SWEEP_DEG / 2.0;
                moon.elevation_deg = PEAK_ELEVATION_DEG;
                moon.timescale = 0.0;
            }
            Self::PoseRise => {
                moon.azimuth_deg = RISE_AZIMUTH_DEG;
                moon.elevation_deg = ELEVATION_FLOOR_DEG;
                moon.timescale = 0.0;
            }
        }
    }
}

/// The motion accumulators, separate from [`Moon`] so they run in f64:
/// frame-sized increments on a f32 stall outright (at 60 fps a ≲5× timescale's
/// phase increment rounds to zero ulps) or accumulate double-digit rate bias.
/// A manual knob write can't teleport the clock.
#[derive(Resource, Default)]
struct MoonClock {
    /// Sim seconds along the traversal path since boot.
    t: f64,
    /// Phase accumulator — the f64 twin of [`Moon::phase`].
    phase: f64,
    /// What `phase` last projected into [`Moon::phase`]: a mismatch means a
    /// manual write landed since, which re-seeds the accumulator (the
    /// manual-write-persists-as-an-offset contract).
    projected: f32,
}

/// Where the back-and-forth traversal puts the moon `t` sim-seconds after
/// boot. The moon crosses half the sky in half a day (rise-to-"set", like a
/// real moon's night), then — to skip daytime — travels back along the same
/// arc instead of setting: azimuth is a day-period triangle wave, elevation an
/// arc from the floor up to the peak and back within each crossing, so it
/// never leaves `[ELEVATION_FLOOR_DEG, PEAK_ELEVATION_DEG]`.
fn traversal_angles(t: f64) -> (f32, f32) {
    let day = (t / DAY_SECONDS).rem_euclid(1.0);
    let s = (1.0 - (2.0 * day - 1.0).abs()) as f32;
    let az = RISE_AZIMUTH_DEG + TRAVERSAL_SWEEP_DEG * s;
    let el = ELEVATION_FLOOR_DEG
        + (PEAK_ELEVATION_DEG - ELEVATION_FLOOR_DEG) * (std::f32::consts::PI * s).sin();
    (az, el)
}

/// One motion tick: advance the clock by `real_dt` scaled by the timescale
/// knob, sweep the phase realistically (one synodic cycle per
/// [`SYNODIC_SECONDS`] of sim time — from the f64 accumulator, re-seeded by
/// any manual phase write so the write persists as an offset), and put the
/// moon on the traversal path. Pure so the motion contract is testable
/// without an `App`.
fn step(moon: &mut Moon, clock: &mut MoonClock, real_dt: f64) {
    if moon.timescale == 0.0 {
        return; // Frozen means FROZEN: a manual pose must survive, not snap to the path.
    }
    let dt = real_dt * moon.timescale as f64;
    clock.t += dt;
    if moon.phase != clock.projected {
        clock.phase = moon.phase as f64;
    }
    clock.phase = (clock.phase + dt / SYNODIC_SECONDS).rem_euclid(1.0);
    moon.phase = clock.phase as f32;
    clock.projected = moon.phase;
    (moon.azimuth_deg, moon.elevation_deg) = traversal_angles(clock.t);
}

/// Runs [`drive_moon_motion`]; the light and sky-uniform syncs order after it
/// so a frame's disc and light see that same frame's motion.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoonMotionSet;

fn drive_moon_motion(time: Res<Time>, mut clock: ResMut<MoonClock>, mut moon: ResMut<Moon>) {
    // step() gates on timescale itself; this outer read-only check keeps a frozen
    // sky from tripping change detection (DerefMut marks the resource changed).
    if moon.timescale == 0.0 {
        return;
    }
    step(&mut moon, &mut clock, time.delta_secs_f64());
}

/// Marks the one moon-driven [`DirectionalLight`].
#[derive(Component)]
struct MoonLight;

/// Spawns the moon's directional light and keeps it tracking the [`Moon`]
/// resource, whose boot value it carries (a plugin field, so no surface can add
/// the moon and forget its knobs). Added by `NightSkyPlugin` — every surface
/// with a sky has the moon, and only those.
pub struct MoonPlugin {
    pub moon: Moon,
}

impl Plugin for MoonPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.moon)
            .init_resource::<MoonClock>()
            .add_systems(Startup, spawn_moon_light)
            .add_systems(
                Update,
                (
                    drive_moon_motion.in_set(MoonMotionSet),
                    sync_moon_light.after(MoonMotionSet),
                ),
            );
    }
}

/// Structure only — every Moon-derived value (illuminance, color, direction)
/// comes from [`sync_moon_light`], the ONE Moon→light mapping, which runs
/// before the first render (the just-inserted resource is `is_changed`).
fn spawn_moon_light(mut commands: Commands) {
    // Cascades stretched from the ~150 m default to mountain scale: the vista
    // world's 30 m grid pitch makes coarse far cascades invisible (rl#281 st 3).
    commands.spawn((
        MoonLight,
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        bevy::light::CascadeShadowConfigBuilder {
            maximum_distance: 9000.0,
            first_cascade_far_bound: 20.0,
            ..default()
        }
        .build(),
    ));
}

fn sync_moon_light(
    moon: Res<Moon>,
    mut lights: Query<(&mut DirectionalLight, &mut Transform), With<MoonLight>>,
) {
    if !moon.is_changed() {
        return;
    }
    for (mut light, mut transform) in &mut lights {
        light.illuminance = moon.illuminance();
        light.color = moon.color();
        *transform = Transform::default().looking_to(-moon.direction(), Vec3::Y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default moon must reproduce the pre-moon static light it replaced:
    /// same color, same illuminance, same direction (rl#374 — the "old light
    /// path deleted in the same change" contract).
    #[test]
    fn default_moon_reproduces_the_old_light() {
        let moon = Moon::default();
        let c = moon.color().to_srgba();
        for (got, want) in [(c.red, 0.85), (c.green, 0.90), (c.blue, 1.0)] {
            assert!((got - want).abs() < 5e-3, "color {got} != {want}");
        }
        assert!((moon.illuminance() - FULL_MOON_LUX).abs() < 1.0);
        let old_forward = Quat::from_euler(EulerRot::XYZ, -0.5, 0.7, 0.0) * Vec3::NEG_Z;
        assert!(
            moon.direction().dot(-old_forward) > 0.9999,
            "default direction {:?} drifted from the old light's {:?}",
            moon.direction(),
            -old_forward
        );
    }

    /// Phase endpoints: new moon is (nearly) dark, full moon is the anchor, and
    /// a quarter moon sits an order of magnitude under full.
    #[test]
    fn phase_luminosity_curve() {
        let m = |phase| Moon { phase, ..default() };
        assert!(m(0.0).illuminance() < 0.03 * FULL_MOON_LUX);
        assert!((m(0.5).illuminance() - FULL_MOON_LUX).abs() < 1.0);
        let quarter = m(0.25).illuminance();
        assert!(
            quarter > 0.05 * FULL_MOON_LUX && quarter < 0.2 * FULL_MOON_LUX,
            "quarter moon {quarter}"
        );
    }

    /// The traversal never sets and never leaves the elevation floor/peak
    /// band, across several days of sim time (the epic's never-setting
    /// contract). Azimuth stays within the one back-and-forth sweep.
    #[test]
    fn traversal_never_sets() {
        let mut t = 0.0;
        while t < 3.0 * DAY_SECONDS {
            let (az, el) = traversal_angles(t);
            assert!(
                (ELEVATION_FLOOR_DEG - 1e-3..=PEAK_ELEVATION_DEG + 1e-3).contains(&el),
                "elevation {el} out of band at t={t}"
            );
            assert!(
                (RISE_AZIMUTH_DEG..=RISE_AZIMUTH_DEG + TRAVERSAL_SWEEP_DEG).contains(&az),
                "azimuth {az} outside the sweep at t={t}"
            );
            t += 97.0; // Prime-ish stride so samples don't alias the path's period.
        }
    }

    /// Boot continuity: at t=0 the path stands exactly on the default pose, so
    /// switching motion on does not jump the shipped stage-1 look.
    #[test]
    fn traversal_starts_at_the_default_pose() {
        let moon = Moon::default();
        assert_eq!(
            traversal_angles(0.0),
            (moon.azimuth_deg, moon.elevation_deg)
        );
    }

    /// Back-and-forth: the return crossing retraces the outbound arc — the
    /// pose at `DAY - t` mirrors the pose at `t`, and mid-crossing is the far
    /// end of the sweep at the floor.
    #[test]
    fn traversal_is_a_ping_pong() {
        for t in [0.1, 0.25, 0.4] {
            let t = t * DAY_SECONDS;
            assert_eq!(traversal_angles(t), traversal_angles(DAY_SECONDS - t));
        }
        let (az, el) = traversal_angles(0.5 * DAY_SECONDS);
        assert!((az - (RISE_AZIMUTH_DEG + TRAVERSAL_SWEEP_DEG)).abs() < 1e-3);
        assert!((el - ELEVATION_FLOOR_DEG).abs() < 1e-3);
    }

    /// The phase sweep is realistic-but-accelerated: at the default 24×
    /// timescale, one real hour advances the phase by 24 h / synodic month.
    #[test]
    fn phase_sweeps_at_the_synodic_rate() {
        let mut moon = Moon::default();
        let mut clock = MoonClock::default();
        step(&mut moon, &mut clock, 3600.0);
        let want = 0.5 + (24.0 * 3600.0 / SYNODIC_SECONDS) as f32;
        assert!(
            (moon.phase - want).abs() < 1e-5,
            "phase {} != {want}",
            moon.phase
        );
        assert_eq!(clock.t, 24.0 * 3600.0);
    }

    /// Frame-sized increments must actually accumulate: an f32 phase
    /// accumulator stalls at low timescales (a realistic-rate increment
    /// rounds to zero f32 ulps near phase 0.5) — the f64 accumulator in
    /// [`MoonClock`] is the fix, and this pins its worst case: one real
    /// minute of 60 fps frames at timescale 1.
    #[test]
    fn phase_accumulates_frame_sized_steps() {
        let mut moon = Moon {
            timescale: 1.0,
            ..default()
        };
        let mut clock = MoonClock::default();
        for _ in 0..3600 {
            step(&mut moon, &mut clock, 1.0 / 60.0);
        }
        let want = 0.5 + (60.0 / SYNODIC_SECONDS) as f32;
        assert!(
            (moon.phase - want).abs() < 1e-6,
            "phase {} != {want}",
            moon.phase
        );
    }

    /// A manual phase write mid-flight persists as an offset: the f64
    /// accumulator re-seeds from the written value instead of overwriting it
    /// with its own history.
    #[test]
    fn manual_phase_write_persists_as_offset() {
        let mut moon = Moon::default();
        let mut clock = MoonClock::default();
        step(&mut moon, &mut clock, 60.0);
        moon.phase = 0.25; // A song poses the phase.
        step(&mut moon, &mut clock, 60.0);
        assert!(
            moon.phase > 0.25 && moon.phase < 0.26,
            "phase {} lost the manual write",
            moon.phase
        );
    }

    /// Timescale 0 freezes the sky completely — the manual-pose escape hatch.
    #[test]
    fn timescale_zero_freezes() {
        let mut moon = Moon {
            timescale: 0.0,
            azimuth_deg: 100.0,
            elevation_deg: 40.0,
            ..default()
        };
        let mut clock = MoonClock::default();
        step(&mut moon, &mut clock, 3600.0);
        assert_eq!(
            moon,
            Moon {
                timescale: 0.0,
                azimuth_deg: 100.0,
                elevation_deg: 40.0,
                ..default()
            }
        );
        assert_eq!(clock.t, 0.0);
    }

    /// Every song lands the moon in a valid, distinct state: phases in [0,1),
    /// hues on the wheel, poses inside the traversal's own band (the elevation
    /// floor contract extends to posed moons), and pose/freeze songs actually
    /// hold — the next motion step must not move a posed moon.
    #[test]
    fn songs_pose_valid_holdable_states() {
        for song in MoonSong::ALL {
            let mut moon = Moon::default();
            song.apply(&mut moon);
            assert!((0.0..1.0).contains(&moon.phase), "{song:?} phase");
            assert!((0.0..360.0).contains(&moon.hue_deg), "{song:?} hue");
            assert!(
                (ELEVATION_FLOOR_DEG..=PEAK_ELEVATION_DEG).contains(&moon.elevation_deg),
                "{song:?} breaks the elevation band: {}",
                moon.elevation_deg
            );
            if moon.timescale == 0.0 {
                let posed = moon;
                step(&mut moon, &mut MoonClock::default(), 3600.0);
                assert_eq!(moon, posed, "{song:?} must hold its pose");
            }
        }
    }

    /// Songs are distinguishable: no two songs map the same moon to the same
    /// state (two identical songs would be two codes for one command). The base
    /// is deliberately off-default everywhere — from a default base the
    /// default-RESTORING songs (silver, drift) are legitimately no-ops.
    #[test]
    fn songs_are_distinct() {
        let base = Moon {
            azimuth_deg: 100.0,
            elevation_deg: 40.0,
            hue_deg: 300.0,
            phase: 0.9,
            timescale: 7.0,
        };
        let states: Vec<Moon> = MoonSong::ALL
            .iter()
            .map(|s| {
                let mut m = base;
                s.apply(&mut m);
                m
            })
            .collect();
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate().skip(i + 1) {
                assert_ne!(
                    a,
                    b,
                    "{:?} and {:?} pose the same moon",
                    MoonSong::ALL[i],
                    MoonSong::ALL[j]
                );
            }
        }
    }

    /// The default-restoring songs really restore the default: silver is the
    /// boot hue, drift the boot tempo, rise the boot pose (plus freeze — the
    /// pose has to hold).
    #[test]
    fn restoring_songs_land_on_the_default() {
        let d = Moon::default();
        let apply = |s: MoonSong| {
            let mut m = Moon::default();
            s.apply(&mut m);
            m
        };
        assert_eq!(apply(MoonSong::HueSilver).hue_deg, d.hue_deg);
        assert_eq!(apply(MoonSong::TempoDrift).timescale, d.timescale);
        let rise = apply(MoonSong::PoseRise);
        assert_eq!(
            (rise.azimuth_deg, rise.elevation_deg),
            (d.azimuth_deg, d.elevation_deg)
        );
        // Zenith is the traversal's own high point — mid-sweep at the peak.
        let zen = apply(MoonSong::PoseZenith);
        let (az, el) = traversal_angles(0.25 * DAY_SECONDS);
        assert!(
            (zen.azimuth_deg - az).abs() < 1e-3,
            "{} != {az}",
            zen.azimuth_deg
        );
        assert!(
            (zen.elevation_deg - el).abs() < 1e-3,
            "{} != {el}",
            zen.elevation_deg
        );
    }

    /// Hue wraps rather than panicking or saturating.
    #[test]
    fn hue_wraps() {
        let a = Moon {
            hue_deg: 400.0,
            ..default()
        };
        let b = Moon {
            hue_deg: 40.0,
            ..default()
        };
        assert_eq!(a.color(), b.color());
    }
}
