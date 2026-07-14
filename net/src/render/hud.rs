use super::driver::{GameState, LocalVehicle};
use super::*;

pub(super) fn sync_controls_context(
    vehicle: Res<LocalVehicle>,
    mut ctx: ResMut<ActiveContext<GcrControls>>,
) {
    ActiveContext::sync(&mut ctx, vehicle.context());
}

/// The not-Playing twin of [`sync_controls_context`]: menu and lobby (Connecting) share
/// the one Menu context (rl#117).
pub(super) fn sync_menu_controls_context(mut ctx: ResMut<ActiveContext<GcrControls>>) {
    ActiveContext::sync(&mut ctx, GcrContext::Menu);
}

#[derive(Component)]
pub(super) struct StatusHud;

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

pub(super) fn update_hud(state: NonSend<GameState>, mut hud: Query<&mut Text, With<StatusHud>>) {
    let Ok(mut text) = hud.single_mut() else {
        return;
    };
    let sim = state.client.sim();
    let status = sim
        .player(state.client.me())
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
    **text =
        format!("You: {status}   |   reach the green pillar, extract - dodge the crab{outcome}",);
}
