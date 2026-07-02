//! HUD: the status line and the controls-overlay context sync. Pure client UI; reads the
//! sim/vehicle state read-only and never writes it.

use super::driver::{GameState, LocalVehicle};
use super::*;

/// Keep the controls overlay's [`ActiveContext`] in sync with the live [`LocalVehicle`], so
/// the on-screen legend + context name follow enter/exit-vehicle automatically. Pure client
/// UI — it never reads or writes the deterministic sim. Cheap (a resource compare), and the
/// one place vehicle state drives the HUD context, so the two can't drift.
pub(super) fn sync_controls_context(
    vehicle: Res<LocalVehicle>,
    mut ctx: ResMut<ActiveContext<GcrControls>>,
) {
    let want = vehicle.context();
    if ctx.0 != want {
        ctx.0 = want;
    }
}

/// The HUD status line (local Alive/Downed/Extracted + the round outcome).
#[derive(Component)]
pub(super) struct StatusHud;

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

pub(super) fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        Text::new("…"),
        TextFont {
            font_size: 22.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(14.0),
            left: Val::Px(14.0),
            ..default()
        },
        StatusHud,
    ));
}

/// Update the HUD: the local player's status, and the round outcome once decided.
pub(super) fn update_hud(state: NonSend<GameState>, mut hud: Query<&mut Text, With<StatusHud>>) {
    let Ok(mut text) = hud.single_mut() else {
        return;
    };
    let sim = state.ls.sim();
    let status = sim
        .player(state.ls.me())
        .map(|p| match p.status() {
            PlayerStatus::Alive => "ALIVE",
            PlayerStatus::Downed => "DOWNED",
            PlayerStatus::Extracted => "EXTRACTED",
        })
        .unwrap_or("—");
    let outcome = match sim.outcome() {
        Outcome::Ongoing => String::new(),
        Outcome::Extracted => "\nROUND WON — extracted!".to_string(),
        Outcome::Wiped => "\nROUND LOST — wiped".to_string(),
    };
    // Status + the one-line objective only. The control bindings are NOT duplicated here:
    // they live in the hold-to-reveal overlay + corner hint (the controls UI), which derive
    // from the one control map — so there's a single on-screen source for them.
    **text =
        format!("You: {status}   |   reach the green pillar, extract - dodge the crab{outcome}",);
}
