//! Build provenance baked at compile time (commit sha + UTC build time) and a subtle
//! always-on corner overlay that shows it. The point is staleness-at-a-glance: a binary
//! that didn't redeploy shows an old sha/date, so a stale install is obvious without
//! anyone having to ask. The values are set by rl-core's build.rs.

/// Short commit the binary was built from; a trailing `+` marks an uncommitted tree.
pub const SHA: &str = env!("RL_BUILD_SHA");
/// UTC build time, `YYYY-MM-DD HH:MM UTC`.
pub const DATE: &str = env!("RL_BUILD_DATE");

/// The one-line corner label, `"<sha> · <date>"`.
pub fn label() -> String {
    format!("{SHA} · {DATE}")
}

/// Spawn the subtle build-stamp in the bottom-right corner. One implementation shared by
/// the game and the demo (each registers it in its own startup schedule), so both stamps
/// are identical. It's static — no update system, hence no marker component. Bottom-right
/// is the one screen corner neither app's HUD/controls-hint already occupies.
#[cfg(feature = "render")]
pub fn spawn_build_info_overlay(mut commands: bevy::prelude::Commands) {
    use bevy::prelude::*;
    commands.spawn((
        Text::new(label()),
        TextFont {
            font_size: 11.0,
            ..default()
        },
        // Dim and semi-transparent: legible if you look for it, ignorable otherwise.
        TextColor(Color::srgba(0.75, 0.75, 0.75, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(6.0),
            right: Val::Px(8.0),
            ..default()
        },
    ));
}
