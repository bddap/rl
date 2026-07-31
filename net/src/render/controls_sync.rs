use super::driver::LocalVehicle;
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
