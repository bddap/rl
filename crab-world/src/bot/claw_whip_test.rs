use super::body::CrabJoint;
use super::headless::tick;

/// The rl#347 flail brake actually lands on the multibody: after spawn,
/// every crab articulation's joint dof carries [`CrabJointId::drive_damping`],
/// not rapier's 0.1 default. Guards the `set_flail_damping` wiring (system
/// ordering, `Added` detection, the `damping_mut` row fill) — a silent
/// miss would leave the plant on default damping with no error.
#[test]
fn flail_damping_lands_on_every_articulation() {
    use bevy_rapier3d::plugin::context::RapierContextJoints;
    use bevy_rapier3d::prelude::RapierMultibodyJointHandle;

    let mut app = super::headless::flat_headless_app();
    tick(&mut app, 3); // spawn + rapier sync + set_flail_damping

    let mut q = app
        .world_mut()
        .query::<(&RapierMultibodyJointHandle, &CrabJoint)>();
    let joints: Vec<_> = q.iter(app.world()).map(|(h, j)| (h.0, j.id)).collect();
    assert!(
        joints.len() > 30,
        "expected a whole crab, got {}",
        joints.len()
    );

    let mut ctx = app.world_mut().query::<&mut RapierContextJoints>();
    let mut ctx = ctx.single_mut(app.world_mut()).expect("one rapier context");
    for (handle, id) in joints {
        let (mb, link_id) = ctx
            .multibody_joints
            .get_mut(handle)
            .expect("every crab joint is a multibody joint");
        let link = mb.link(link_id).expect("handle names a live link");
        let (assembly_id, ndofs) = (link.assembly_id(), link.joint().ndofs());
        let damping: Vec<f32> = mb.damping().as_slice()[assembly_id..assembly_id + ndofs].to_vec();
        assert_eq!(
            damping,
            &[id.drive_damping()][..],
            "{id:?}: flail-brake damping did not land on the multibody dof"
        );
    }
}
