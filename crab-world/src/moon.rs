//! The moon — THE light source (rl#374). One knob-driven state drives both the
//! sky's visible disc (`sky.wgsl`, via `StarSkyMaterial` uniforms) and the one
//! `DirectionalLight`, so the light on the terrain and the disc in the sky can
//! never disagree. There is no other directional light anywhere.

use bevy::prelude::*;

/// Full-moon illuminance (lux, stylized — real moonlight is ~0.3 lux; this is the
/// vista look's tuned brightness, carried over from the pre-moon static light).
const FULL_MOON_LUX: f32 = 9500.0;

/// The hue knob turns only the hue of an otherwise fixed pastel: saturation 1.0 /
/// lightness 0.925 in HSL, chosen so the default hue reproduces the pre-moon
/// light color exactly (`hsl(220°) == srgb(0.85, 0.90, 1.0)` — guarded by test).
const MOON_SATURATION: f32 = 1.0;
const MOON_LIGHTNESS: f32 = 0.925;

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
}

impl Default for Moon {
    /// Reproduces the pre-moon static light: its euler-angle transform pointed
    /// the light along -(0.644, 0.367, 0.671), its color was `hsl(220°)`, and a
    /// full moon carries the same illuminance.
    fn default() -> Self {
        Self {
            azimuth_deg: 43.8,
            elevation_deg: 21.5,
            hue_deg: 220.0,
            phase: 0.5,
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
    /// dark but not a void (ambient starlight stays separate).
    pub fn illuminance(&self) -> f32 {
        let f = self.illuminated_fraction();
        FULL_MOON_LUX * (0.02 + 0.98 * f.powi(3))
    }
}

/// Marks the one moon-driven [`DirectionalLight`].
#[derive(Component)]
struct MoonLight;

/// Spawns the moon's directional light and keeps it tracking the [`Moon`]
/// resource. Added by `NightSkyPlugin` — every surface with a sky has the moon,
/// and only those.
pub struct MoonPlugin;

impl Plugin for MoonPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Moon>()
            .add_systems(Startup, spawn_moon_light)
            .add_systems(Update, sync_moon_light);
    }
}

fn spawn_moon_light(mut commands: Commands, moon: Res<Moon>) {
    // Cascades stretched from the ~150 m default to mountain scale: the vista
    // world's 30 m grid pitch makes coarse far cascades invisible (rl#281 st 3).
    commands.spawn((
        MoonLight,
        DirectionalLight {
            shadows_enabled: true,
            illuminance: moon.illuminance(),
            color: moon.color(),
            ..default()
        },
        bevy::light::CascadeShadowConfigBuilder {
            maximum_distance: 9000.0,
            first_cascade_far_bound: 20.0,
            ..default()
        }
        .build(),
        Transform::default().looking_to(-moon.direction(), Vec3::Y),
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
