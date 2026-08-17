//! rl#377 diagnostic probe (scratch, not a gate): park a plane in assorted spots /
//! residual-throttle settings on the real GCR terrain and print per-tick pose
//! motion after settle. Run: `cargo test -p crab-world --test park_probe -- --nocapture --ignored`

use bevy::prelude::*;
use crab_world::bot::headless::{headless_app, tick};
use crab_world::vehicle::{
    Boarding, PilotCommand, PilotId, VehicleControls, VehicleKind, VehiclePlugin, clear_of_ground,
};

fn park_run(label: &str, xz: Vec2, throttle_trim_ticks: u32) {
    let mut app = headless_app();
    app.add_plugins(VehiclePlugin);
    let terrain = app
        .world()
        .resource::<crab_world::terrain::Terrain>()
        .clone();
    let pos = clear_of_ground(Vec3::new(xz.x, -10_000.0, xz.y), 0.01, &terrain);
    app.world_mut().resource_mut::<VehicleControls>().0.insert(
        PilotId(0),
        PilotCommand::new(
            VehicleKind::Plane,
            Boarding {
                pos,
                yaw: 0.0,
                velocity: Vec3::ZERO,
            },
        ),
    );
    // Trim throttle up for the requested ticks (rt full = +1 trim), then release.
    for _ in 0..throttle_trim_ticks {
        {
            let mut c = app.world_mut().resource_mut::<VehicleControls>();
            c.0.get_mut(&PilotId(0)).unwrap().throttle_trim = 1.0;
        }
        tick(&mut app, 1);
    }
    {
        let mut c = app.world_mut().resource_mut::<VehicleControls>();
        c.0.get_mut(&PilotId(0)).unwrap().throttle_trim = 0.0;
    }
    // Settle.
    tick(&mut app, 300);
    // Observe 300 ticks: per-tick translation delta + rotation angle delta.
    let mut prev: Option<(Vec3, Quat)> = None;
    let mut max_d = 0.0f32;
    let mut max_rot = 0.0f32;
    let mut moved_ticks = 0u32;
    let mut sum_d = 0.0f32;
    let mut start = Vec3::ZERO;
    let mut end = Vec3::ZERO;
    let mut max_angvel = 0.0f32;
    let mut awake_ticks = 0u32;
    for i in 0..300 {
        tick(&mut app, 1);
        let sleeping = {
            use bevy_rapier3d::plugin::context::RapierRigidBodySet;
            use bevy_rapier3d::prelude::RapierRigidBodyHandle;
            let handle = {
                let mut hq = app
                    .world_mut()
                    .query::<(&crab_world::vehicle::Vehicle, &RapierRigidBodyHandle)>();
                hq.single(app.world()).expect("one craft").1.0
            };
            let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
            let set = set_q.single(app.world()).expect("one body set");
            set.bodies
                .get(handle)
                .map(|b| b.is_sleeping())
                .unwrap_or(false)
        };
        if !sleeping {
            awake_ticks += 1;
        }
        let mut q = app.world_mut().query::<(
            &crab_world::vehicle::Vehicle,
            &Transform,
            &bevy_rapier3d::prelude::Velocity,
        )>();
        let (_, tf, vel) = q.single(app.world()).expect("one craft");
        let (p, r) = (tf.translation, tf.rotation);
        max_angvel = max_angvel.max(vel.angular.length());
        if i == 0 {
            start = p;
        }
        end = p;
        if let Some((pp, pr)) = prev {
            let d = p.distance(pp);
            let rot = pr.angle_between(r);
            if d > 0.0 || rot > 0.0 {
                moved_ticks += 1;
            }
            max_d = max_d.max(d);
            max_rot = max_rot.max(rot);
            sum_d += d;
        }
        prev = Some((p, r));
    }
    let slope = {
        let e = 0.5;
        let hx = (terrain.height(xz.x + e, xz.y) - terrain.height(xz.x - e, xz.y)) / (2.0 * e);
        let hz = (terrain.height(xz.x, xz.y + e) - terrain.height(xz.x, xz.y - e)) / (2.0 * e);
        (hx * hx + hz * hz).sqrt()
    };
    println!(
        "{label}: pos=({:.1},{:.1},{:.1}) slope={:.3} moved={moved_ticks}/299 max_d={:.3}mm \
         mean_d={:.4}mm max_rot={:.4}deg net={:.3}mm awake={awake_ticks}/300 max_w={max_angvel:.5}rad_s",
        end.x,
        end.y,
        end.z,
        slope,
        max_d * 1e3,
        sum_d / 299.0 * 1e3,
        max_rot.to_degrees(),
        start.distance(end) * 1e3,
    );
}

#[test]
#[ignore]
fn park_sweep() {
    crab_world::bot::headless::pin_single_thread_pools();
    // Near-origin flat-ish, far-coordinate corner (f32 ULP ~1 mm), and a slope hunt.
    park_run("origin t0", Vec2::new(0.0, 0.0), 0);
    park_run("far t0", Vec2::new(-12900.0, 9360.0), 0);
    // Residual throttle: landed without trimming back to zero.
    for trim in [30, 60, 100] {
        park_run(&format!("far trim{trim}"), Vec2::new(-12900.0, 9360.0), trim);
    }
    // Slope spots: sample the grid for progressively steeper parks near the far corner.
    for (dx, dz) in [(50.0, 0.0), (0.0, 80.0), (120.0, 120.0), (-200.0, 40.0)] {
        park_run(
            &format!("far+({dx},{dz})"),
            Vec2::new(-12900.0 + dx, 9360.0 + dz),
            0,
        );
    }
}
