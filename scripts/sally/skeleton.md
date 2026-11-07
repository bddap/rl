# Sally's skeleton

The rig `rig.py` builds for the purchased *Grapsus grapsus* mesh. This file is its
input: `rig.py` parses the three tables below, so a change here is a change to the rig.
No joint ranges have been published for *G. grapsus* itself; each row names the crab
its number was measured on.

## Constants

| constant | value | source |
|---|---|---|
| `carapace_width_m` | 0.053 | Six large male *G. tenuicrustatus*, the congener whose gait sets the MC range: carapace width 53 ± 4 mm (Martinez et al. 1998, Methods). Adult *G. grapsus* reach 69.3 mm (Gutiérrez Rubio et al. 2020, Table 1). |
| `mass_kg` | 0.074 | The same six crabs: 74 ± 10 g. Every body's density is this mass over the total hull volume. |
| `skin_blend_rings` | 2 | Edge rings over which a vertex's weight blends across a joint; the cuticle is rigid, so the blend stays at the seam. |

## Segments

A crab's walking leg has six segments, "since the ischium and basis are fused"
(Vidal-Gadea et al. 2008, Fig. 6, of *Carcinus maenas* and *Libinia emarginata*). No
source states the fusion for the cheliped; it is taken to match. Mesh regions are the purchased mesh's own vertex groups
(`Def_<prefix>.<region>.<side>`); every vertex outside a listed region belongs to the body.

| appendage | segment | mesh regions |
|---|---|---|
| walking leg | coxa | `000b` |
| walking leg | basi-ischium | `001` |
| walking leg | merus | `002` |
| walking leg | carpus | `003` |
| walking leg | propodus | `004` |
| walking leg | dactylus | `005` |
| cheliped | coxa | `000a` |
| cheliped | basi-ischium | `000` `001` |
| cheliped | merus | `002` |
| cheliped | carpus | `003` `004` |
| cheliped | propodus | `005` `006b` |
| cheliped | dactylus | `006` |

## Joints

Each joint is a single-axis hinge: crab leg joints "are uniplanar" (Vidal-Gadea et al.
2008, §3.6). The plane column is that paper's: TC and CP swing the leg fore and aft
(frontal plane), CB, MC and PD raise and lower it in its own vertical plane
(transverse), and BM turns the leg's long axis forward on a diagonal between the two
(oblique). The paper says the diagonal of *Libinia*, "and to a lesser extent" of
*Carcinus*; the 45° the rig gives it is this table's choice. The cheliped's PD opens the movable finger in the plane of
both fingers (chela).

An angle is the segment's deflection from straight continuation of its parent (from
the body's lateral axis for TC). Positive is dorsal for transverse joints, anterior for
frontal and oblique ones, and opening for the chela. A range written `a..b` is in that
convention. A bare extent is a passive range whose endpoints the sources do not give
in this convention; the rig centres it on the purchased mesh's rest angle, which
`rig.py` measures and reports. That centring is an assumption, not anatomy: it gives
the walking-leg dactylus as much dorsal as ventral travel.

| appendage | joint | segment | plane | range° | source |
|---|---|---|---|---|---|
| walking leg | TC | coxa | frontal | 123 | *Carcinus maenas*, passive, 123 ± 24 (Vidal-Gadea et al. 2008, Table 3) |
| walking leg | CB | basi-ischium | transverse | 138 | *C. maenas*, passive, 138 ± 15 (Vidal-Gadea et al. 2008, Table 3) |
| walking leg | BM | merus | oblique | 26 | *C. maenas*, passive, 26 ± 6 (Vidal-Gadea et al. 2008, Table 3) |
| walking leg | MC | carpus | transverse | -148..-25 | *C. maenas* passive extent 123 ± 22 (Vidal-Gadea et al. 2008, Table 3), centred on the 68–119° *G. tenuicrustatus* sweeps walking in air (Martinez et al. 1998, Table 6), an angle at the MC joint between the IM and PD joints, so it folds in CP flexion |
| walking leg | CP | propodus | frontal | 79 | *C. maenas*, passive, 79 ± 11 (Vidal-Gadea et al. 2008, Table 3) |
| walking leg | PD | dactylus | transverse | 151 | *C. maenas*, passive, 151 ± 11 (Vidal-Gadea et al. 2008, Table 3) |
| cheliped | TC | coxa | frontal | 123 | none for a cheliped; the walking leg's |
| cheliped | CB | basi-ischium | transverse | 138 | none for a cheliped; the walking leg's |
| cheliped | BM | merus | oblique | 26 | none for a cheliped; the walking leg's |
| cheliped | MC | carpus | transverse | 123 | none for a cheliped; the walking leg's passive extent |
| cheliped | CP | propodus | frontal | 74 | *Cancer magister*, isolated cheliped, +37° producted to −37° reducted about the paper's own extension zero, which is not this table's straight line; its text calls the span "about 65°" (Cooper & Hartman 1994) |
| cheliped | PD | dactylus | chela | 0..45 | none found; closed at 0, opened to 45° |

## References

- Cooper, R. L. & Hartman, H. B. (1994). Responses of bender apodeme tension receptors
  in the Dungeness crab, *Cancer magister*. *Comp. Biochem. Physiol. A* 109, 479–486.
  https://doi.org/10.1016/0300-9629(94)90153-8
- Gutiérrez Rubio, Y. et al. (2020). Size composition and fecundity of the rock crab
  *Grapsus grapsus* (Linnaeus, 1758) in the islands of Navachiste Bay, Sinaloa, Mexico.
  *Indian J. Fish.* 67(1). https://doi.org/10.21077/ijf.2019.67.1.87526-17
- Martinez, M. M., Full, R. J. & Koehl, M. A. R. (1998). Underwater punting by an
  intertidal crab: a novel gait revealed by the kinematics of pedestrian locomotion in
  air versus water. *J. Exp. Biol.* 201, 2609–2623. https://doi.org/10.1242/jeb.201.18.2609
- Vidal-Gadea, A. G., Rinehart, M. D. & Belanger, J. H. (2008). Skeletal adaptations for
  forwards and sideways walking in three species of decapod crustaceans. *Arthropod
  Struct. Dev.* 37, 95–108. https://doi.org/10.1016/j.asd.2007.06.002
