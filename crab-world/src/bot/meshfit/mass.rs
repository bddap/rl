//! Analytic capsule mass properties — closed-form ground truth, no rapier collider.

/// Rigid-body mass properties of a capsule, computed analytically (cylinder +
/// two hemispheres) so mass derives from the dimensions without constructing a
/// rapier collider. `i_axial` is the moment about the capsule's own long axis;
/// `i_perp` about a transverse axis through the centre of mass.
//
// Test-only: the analytic mass formula has no production caller — the live body
// lets rapier compute inertia from the spawned collider + density — but it is the
// independent ground truth `capsule_reduces_to_sphere` checks the closed forms
// against, so it stays gated to the test build rather than shipping unused.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub struct CapsuleMass {
    pub mass: f32,
    pub i_axial: f32,
    pub i_perp: f32,
}

/// Analytic capsule inertia about its centre of mass. `half_height` is the
/// cylinder half-length (caps excluded), matching `capsule_y`. Standard
/// solid-capsule formulae (uniform density ρ): the cylinder contributes its
/// usual terms; each hemisphere its own plus a parallel-axis shift for the
/// transverse moment.
#[cfg(test)]
pub fn capsule_mass(radius: f32, half_height: f32, density: f32) -> CapsuleMass {
    let r = radius as f64;
    let h = (half_height * 2.0) as f64; // full cylinder length
    let rho = density as f64;
    let pi = std::f64::consts::PI;

    let m_cyl = rho * pi * r * r * h;
    let m_hemi = rho * (2.0 / 3.0) * pi * r * r * r; // one hemisphere
    let mass = m_cyl + 2.0 * m_hemi;

    // Axial (about the long axis): cylinder ½ m r², two hemispheres ⅖ m_hemi r² each.
    let i_axial = 0.5 * m_cyl * r * r + 2.0 * (2.0 / 5.0) * m_hemi * r * r;

    // Transverse: cylinder (1/12 m (3r² + h²)); each hemisphere has I about its
    // own centroid plus a parallel-axis shift to the capsule centre. Hemisphere
    // centroid sits 3r/8 from its flat face; the flat face is at h/2 from centre.
    let i_cyl_perp = m_cyl * (3.0 * r * r + h * h) / 12.0;
    let i_hemi_own = (2.0 / 5.0) * m_hemi * r * r; // sphere's ⅖mr² split per hemi
    // Parallel-axis: distance from capsule centre to hemisphere centroid.
    let d = h / 2.0 + 3.0 * r / 8.0;
    // Subtract the self-term already at the hemisphere centroid vs its flat face.
    // Using the standard composite result: I_hemi_perp(about capsule centre) =
    //   (2/5)m r²  - m (3r/8)²            [shift own-axis to centroid]
    //   + m (h/2 + 3r/8)²                 [shift centroid to capsule centre]
    let i_hemi_perp = i_hemi_own - m_hemi * (3.0 * r / 8.0).powi(2) + m_hemi * d * d;
    let i_perp = i_cyl_perp + 2.0 * i_hemi_perp;

    CapsuleMass {
        mass: mass as f32,
        i_axial: i_axial as f32,
        i_perp: i_perp as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Capsule mass formula sanity: a capsule with zero half-height is a sphere,
    /// so its mass and (isotropic) inertia must match the sphere closed forms.
    #[test]
    fn capsule_reduces_to_sphere() {
        let r = 0.1f32;
        let rho = 3.0f32;
        let m = capsule_mass(r, 0.0, rho);
        let sphere_mass = rho * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3);
        let sphere_i = 0.4 * sphere_mass * r * r;
        assert!(
            (m.mass - sphere_mass).abs() / sphere_mass < 1e-4,
            "mass {m:?}"
        );
        assert!(
            (m.i_axial - sphere_i).abs() / sphere_i < 1e-3,
            "i_axial {m:?}"
        );
        assert!(
            (m.i_perp - sphere_i).abs() / sphere_i < 1e-3,
            "i_perp {m:?}"
        );
        assert!(
            (m.i_axial - m.i_perp).abs() / m.i_axial < 1e-3,
            "sphere must be isotropic: {m:?}"
        );
    }
}
