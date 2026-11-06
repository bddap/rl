//! The host's manual physics pump: the wall-clock fixed-timestep auto-pump is parked
//! ([`park_fixed_auto_pump`]), so [`pump_fixed_steps`] is the ONLY thing that advances
//! `FixedMain`, and the per-tick count comes from [`crate::cadence::steps_for_tick`], the
//! one source for the 64:30 staircase.

use bevy::prelude::*;

pub(crate) fn pump_fixed_steps(world: &mut World, steps: u32) {
    use bevy::app::FixedMain;
    use bevy::time::{Fixed, Virtual};

    if steps == 0 {
        return;
    }
    for _ in 0..steps {
        let fixed = world.resource::<Time<Fixed>>().as_generic();
        *world.resource_mut::<Time>() = fixed;
        world.run_schedule(FixedMain);
    }
    let virt = world.resource::<Time<Virtual>>().as_generic();
    *world.resource_mut::<Time>() = virt;
}

/// Park the wall-clock fixed-timestep auto-pump — EXACTLY: every peer's `FixedUpdate`
/// then advances only through [`pump_fixed_steps`], which only the server-auth arm
/// calls. Two layers because neither alone is airtight: the overstep discard runs
/// every frame BEFORE bevy's accrual+expend, so the accumulator entering the expend
/// holds at most one frame's virtual delta — which the huge timestep can never expend
/// (a bare huge timestep still fired one rogue un-pumped step per ~24 h of accumulated
/// uptime, rl#298 stage 4).
pub(crate) fn park_fixed_auto_pump(app: &mut App) {
    use bevy::app::RunFixedMainLoop;

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
    app.add_systems(
        RunFixedMainLoop,
        discard_parked_overstep.in_set(bevy::app::RunFixedMainLoopSystems::BeforeFixedMainLoop),
    );
}

/// [`park_fixed_auto_pump`]'s per-frame half: zero the fixed accumulator before the
/// frame's accrual. [`pump_fixed_steps`] is untouched — it runs `FixedMain` directly
/// and never consults the accumulator.
fn discard_parked_overstep(mut fixed: ResMut<Time<Fixed>>) {
    let overstep = fixed.overstep();
    fixed.discard_overstep(overstep);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The park is EXACT (rl#298 stage 4): a marathon host accumulates unbounded
    /// wall-clock into `Time<Fixed>`'s overstep, and past the 86400 s timestep the
    /// auto-pump fires one rogue un-pumped step. The discard half must keep the
    /// accumulator from ever reaching the timestep; the control app (a timestep-only
    /// park) proves this test would catch the rogue step.
    #[test]
    fn parked_auto_pump_never_fires_even_past_a_day_of_uptime() {
        use std::time::Duration;

        use bevy::time::{Fixed, TimeUpdateStrategy, Virtual};

        #[derive(Resource, Default)]
        struct FixedRuns(u32);

        let build = |exact_park: bool| {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins);
            app.init_resource::<FixedRuns>()
                .add_systems(FixedUpdate, |mut runs: ResMut<FixedRuns>| runs.0 += 1);
            // Uncapped virtual time so 25 h of uptime is a few updates, not 350k
            // real-delta frames under the 250 ms clamp.
            app.world_mut()
                .resource_mut::<Time<Virtual>>()
                .set_max_delta(Duration::MAX);
            if exact_park {
                park_fixed_auto_pump(&mut app);
            } else {
                app.world_mut()
                    .resource_mut::<Time<Fixed>>()
                    .set_timestep(Duration::from_secs(86_400));
            }
            app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(
                30_000,
            )));
            app
        };

        let mut control = build(false);
        for _ in 0..5 {
            control.update();
        }
        assert!(
            control.world().resource::<FixedRuns>().0 > 0,
            "teeth: 120000 s of uptime must overflow the bare 86400 s park — if it \
             doesn't, this test can no longer detect the rogue step"
        );

        let mut parked = build(true);
        for _ in 0..5 {
            parked.update();
        }
        assert_eq!(
            parked.world().resource::<FixedRuns>().0,
            0,
            "the exact park must never let the auto-pump fire"
        );
        let overstep = parked.world().resource::<Time<Fixed>>().overstep();
        assert!(
            overstep <= Duration::from_secs(30_000),
            "post-frame residual must be at most one frame's delta, got {overstep:?}"
        );
    }
}
