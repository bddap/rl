//! The d-pad combo map (rl#358): a spatial, DISCOVERED-ONLY presentation of the chord
//! code space, shown full-screen while a code is being entered (the held-modifier
//! capture) and gone the frame the hold ends — no linger, no fade-out (rl#381: the
//! player is the bottleneck, not animations). Since the owner's post-playtest pick
//! (2026-08-12) this IS the only code surface — the textual chord menu is gone, and
//! look-and-feel iterates here.
//!
//! The layout is the circular zoom-quadrant space from the winning design submission
//! (`proposal/dpad-map-zoomquad`): every node is a circle, its four children nested
//! inside it in the d-pad directions, so a code's address IS its position and each
//! press dives one circle deeper while ancestors ghost past the edges. Button-true by
//! construction. Owner directives built in:
//! - **Discovered-only**: the map renders codes the player has PLAYED (accepted
//!   resolutions), persisted per save file — no teasers. The satisfaction is watching
//!   the map grow more complex over time.
//! - **Sparse-space rendering** (rl#380): codes are uncapped and the space is sparsely
//!   populated, so fork options are sized by discovered-path density — the sole known
//!   continuation renders large, undiscovered directions are quiet seed rings.
//! - **Text-free as possible**: no header, no legend; the one text concession is a
//!   discovered room's command label once its circle is big enough to read as a place.
//!
//! Rendering is a CPU-painted canvas ([`Canvas`] → `Image` → UI node) plus a pooled set
//! of UI text labels — no egui (the offscreen evidence app has none) and no gizmos (no
//! 2D camera), so the same path draws in the windowed client and in fp-screenshot
//! evidence captures. Geometry runs in f64: node radii shrink exponentially with code
//! depth, and with uncapped codes f32 loses the child offsets to rounding around
//! depth 14.

use std::collections::BTreeSet;
use std::path::PathBuf;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::math::DVec2;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crab_world::chord::{ChordDir, Chords};

use crate::controls::{GCR_CHORDS, GcrControls};

// ---------------------------------------------------------------------------
// Discovered set — the per-save persistent record of played codes.
// ---------------------------------------------------------------------------

const DIRS: [ChordDir; 4] = [
    ChordDir::Up,
    ChordDir::Down,
    ChordDir::Left,
    ChordDir::Right,
];

/// Child-slot index of a direction — `DIRS[dir_index(d)] == d` by construction
/// (`DIRS` lists `ChordDir` in declaration order; tied down by a test).
fn dir_index(d: ChordDir) -> usize {
    d as usize
}

fn code_to_str(code: &[ChordDir]) -> String {
    code.iter()
        .map(|d| match d {
            ChordDir::Up => 'U',
            ChordDir::Down => 'D',
            ChordDir::Left => 'L',
            ChordDir::Right => 'R',
        })
        .collect()
}

/// THE `UDLR`-letters ↔ code codec (save files, CLI tap specs) — the one alphabet;
/// parse callers elsewhere (fp-screenshot) go through this, never a second match.
pub fn code_from_str(s: &str) -> Option<Vec<ChordDir>> {
    s.trim()
        .chars()
        .map(|c| match c {
            'U' => Some(ChordDir::Up),
            'D' => Some(ChordDir::Down),
            'L' => Some(ChordDir::Left),
            'R' => Some(ChordDir::Right),
            _ => None,
        })
        .collect()
}

/// The codes the player has played (accepted resolutions), persisted one code per
/// line (`UDLR` letters). An absent file is a fresh save (empty map), not an error;
/// a malformed line is skipped with a warning — this is user state, not a shipped
/// asset, so the missing-assets panic policy does not apply.
#[derive(Resource)]
pub struct DiscoveredCodes {
    codes: BTreeSet<Vec<ChordDir>>,
    /// The session's replicated discovered set (rl#398) — on a joined client, the
    /// host's map out of the adopted snapshots; on the solo/host arm, its own set
    /// mirrored back through the same seam. Rendered (union with `codes`) but never
    /// persisted: the SAVE stays codes this player PLAYED (the rl#358 directive) —
    /// joining a session shows the party's map, it doesn't grant it.
    remote: BTreeSet<Vec<ChordDir>>,
    file: Option<PathBuf>,
}

impl DiscoveredCodes {
    pub fn load(mut file: Option<PathBuf>) -> Self {
        let mut codes = BTreeSet::new();
        if let Some(path) = &file {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    for line in text.lines().filter(|l| !l.trim().is_empty()) {
                        match code_from_str(line) {
                            Some(code) => {
                                codes.insert(code);
                            }
                            None => {
                                warn!("chord-map save {}: bad line {line:?}", path.display())
                            }
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {} // a fresh save
                Err(e) => {
                    // A transient read failure must NOT masquerade as a fresh save —
                    // the next discovery would rewrite the file from the empty set
                    // and erase every prior one. Run unpersisted instead.
                    warn!(
                        "chord-map save {} unreadable ({e}); running unpersisted",
                        path.display()
                    );
                    file = None;
                }
            }
        }
        Self {
            codes,
            remote: BTreeSet::new(),
            file,
        }
    }

    /// Adopt the session's replicated discovered set (wholesale — each snapshot
    /// carries the full set, so the newest one simply replaces).
    pub fn set_remote(&mut self, remote: BTreeSet<Vec<ChordDir>>) {
        self.remote = remote;
    }

    /// What the map renders: the local save UNION the session's replicated set.
    fn visible(&self) -> BTreeSet<Vec<ChordDir>> {
        self.codes.union(&self.remote).cloned().collect()
    }

    /// Record a played code; returns whether it is NEW (the growth moment). Saves
    /// eagerly — discoveries are rare and the file is tiny.
    fn insert(&mut self, code: &[ChordDir]) -> bool {
        // The empty code (a bare modifier tap) is never a discovery: its room IS the
        // root, which is always on the map — and it can't round-trip the
        // one-code-per-line save format (its line is blank, which `load` skips), so
        // persisting it would re-"discover" it every session. Consequence: if a
        // registry ever assigns the empty code, the root will not bloom/label for
        // it — root discovery needs its own representation first.
        if code.is_empty() || !self.codes.insert(code.to_vec()) {
            return false;
        }
        if let Some(path) = &self.file {
            let body: String = self.codes.iter().map(|c| code_to_str(c) + "\n").collect();
            // Temp-file + rename: an in-place truncate-and-write torn by a crash
            // would drop the tail codes and the NEXT save would make that permanent.
            let tmp = path.with_extension("tmp");
            let write = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&tmp, body))
                .and_then(|()| std::fs::rename(&tmp, path));
            if let Err(e) = write {
                warn!("chord-map save {} failed: {e}", path.display());
            }
        }
        true
    }

    /// The LOCAL save's codes only — what the host stamps onto outgoing snapshots
    /// ([`super::driver`]); replicated codes never relay onward through this peer.
    pub(super) fn codes(&self) -> &BTreeSet<Vec<ChordDir>> {
        &self.codes
    }
}

/// The windowed client's save location: `$RL_CHORD_MAP_FILE`, else
/// [`super::data_dir`] + `discovered-codes.txt`.
pub(crate) fn default_save_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RL_CHORD_MAP_FILE") {
        return Some(PathBuf::from(p));
    }
    Some(super::data_dir()?.join("discovered-codes.txt"))
}

// ---------------------------------------------------------------------------
// The discovered tree.
// ---------------------------------------------------------------------------

struct MapNode {
    code: Vec<ChordDir>,
    children: [Option<usize>; 4],
    /// The registered command's label, shown ONLY once discovered.
    command: Option<&'static str>,
    /// On the live entered path only — not (yet) part of the discovered map.
    provisional: bool,
    /// Discovered (non-provisional) nodes in this subtree, self included — the
    /// rl#380 density weight that sizes fork options.
    weight: usize,
}

struct MapTree {
    nodes: Vec<MapNode>,
}

impl MapTree {
    /// Every prefix of every discovered code, plus the live entered path as
    /// provisional nodes — the focus must exist on the map to zoom to, even while
    /// typing into undiscovered space.
    fn build(discovered: &BTreeSet<Vec<ChordDir>>, entered: &[ChordDir]) -> Self {
        let mut tree = Self {
            nodes: vec![MapNode {
                code: Vec::new(),
                children: [None; 4],
                command: None,
                provisional: true,
                weight: 0,
            }],
        };
        for code in discovered {
            let leaf = tree.insert_path(code);
            tree.nodes[leaf].command = GCR_CHORDS.entry(code).map(|e| e.label);
        }
        // The whole prefix chain of every discovered code is discovered structure;
        // marking parents along insert_path would miss prefixes inserted late.
        let discovered_paths: Vec<usize> =
            discovered.iter().filter_map(|c| tree.index_of(c)).collect();
        for leaf in discovered_paths {
            let mut at = leaf;
            loop {
                tree.nodes[at].provisional = false;
                match tree.parent_of(at) {
                    Some(p) => at = p,
                    None => break,
                }
            }
        }
        tree.insert_path(entered);
        // Weights bottom-up: children have larger indices than parents (insert
        // order), so a reverse scan accumulates each subtree before its parent
        // consumes it.
        for i in (0..tree.nodes.len()).rev() {
            let own = usize::from(!tree.nodes[i].provisional);
            let kids: usize = tree.nodes[i]
                .children
                .iter()
                .flatten()
                .map(|&c| tree.nodes[c].weight)
                .sum();
            tree.nodes[i].weight = own + kids;
        }
        tree
    }

    fn insert_path(&mut self, code: &[ChordDir]) -> usize {
        let mut at = 0;
        for &d in code {
            let slot = dir_index(d);
            at = match self.nodes[at].children[slot] {
                Some(c) => c,
                None => {
                    let mut child_code = self.nodes[at].code.clone();
                    child_code.push(d);
                    self.nodes.push(MapNode {
                        code: child_code,
                        children: [None; 4],
                        command: None,
                        provisional: true,
                        weight: 0,
                    });
                    let c = self.nodes.len() - 1;
                    self.nodes[at].children[slot] = Some(c);
                    c
                }
            };
        }
        at
    }

    fn index_of(&self, code: &[ChordDir]) -> Option<usize> {
        let mut at = 0;
        for &d in code {
            at = self.nodes[at].children[dir_index(d)]?;
        }
        Some(at)
    }

    fn parent_of(&self, n: usize) -> Option<usize> {
        let code = &self.nodes[n].code;
        let (last, prefix) = code.split_last()?;
        let p = self
            .index_of(prefix)
            .expect("every node's prefix chain is in the tree");
        debug_assert_eq!(self.nodes[p].children[dir_index(*last)], Some(n));
        Some(p)
    }
}

// ---------------------------------------------------------------------------
// Circular zoom-quadrant geometry — the winning submission's space.
// ---------------------------------------------------------------------------

/// Canvas-space (y down) unit vector of a button direction.
fn dir_vec(d: ChordDir) -> DVec2 {
    match d {
        ChordDir::Up => DVec2::new(0.0, -1.0),
        ChordDir::Down => DVec2::new(0.0, 1.0),
        ChordDir::Left => DVec2::new(-1.0, 0.0),
        ChordDir::Right => DVec2::new(1.0, 0.0),
    }
}

/// Child circle radius, in parent radii — the submission mock's SHRINK.
const CIRCLE_SHRINK: f64 = 0.36;
/// Child center offset from the parent center, in parent radii (mock STEP). A
/// density-boosted child instead nestles at `1 - radius` so it stays inside the
/// parent circle.
const CIRCLE_STEP: f64 = 0.62;
/// Fork-density size range (rl#380): a child's shrink is scaled by
/// `MIN + SPAN * weight_share`, so the sole known continuation at a fork renders
/// large and thin branches stay small.
const DENSITY_SCALE_MIN: f64 = 0.55;
const DENSITY_SCALE_SPAN: f64 = 0.95;
/// Undiscovered-direction seed rings: fraction of the nominal child radius, and the
/// minimum on-screen radius of the parent before seeds appear at all (quiet space).
const SEED_SHRINK: f64 = 0.40;
const SEED_MIN_PARENT_PX: f64 = 130.0;
/// Radius floor (rl#384): each level multiplies radius by ~0.2–0.54, so ~1150
/// levels underflow f64 to 0, and a zero focus radius NaNs the whole canvas via
/// `ln(0)` → inf scale. Well above subnormals, far below anything visible.
const MIN_RADIUS: f64 = 1e-300;

/// A node circle in world space (root = unit circle at the origin).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Circle {
    center: DVec2,
    radius: f64,
}

/// The rl#380 density scale of each child at a fork: share of the fork's discovered
/// weight, floored so a provisional (live-typed) child still has a visible circle.
fn child_scales(tree: &MapTree, n: usize) -> [f64; 4] {
    let w = |c: Option<usize>| c.map_or(0, |c| tree.nodes[c].weight.max(1));
    let total: usize = tree.nodes[n].children.iter().map(|&c| w(c)).sum();
    let mut scales = [1.0; 4];
    for (slot, &c) in tree.nodes[n].children.iter().enumerate() {
        if c.is_some() && total > 0 {
            scales[slot] = DENSITY_SCALE_MIN + DENSITY_SCALE_SPAN * (w(c) as f64 / total as f64);
        }
    }
    scales
}

/// Every node's circle, root-down. The child offset is capped at `1 - child_radius`
/// parent radii so a density-boosted circle nestles inward instead of poking out of
/// its parent — containment is what makes the address readable.
fn layout_circles(tree: &MapTree) -> Vec<Circle> {
    let mut circles = vec![
        Circle {
            center: DVec2::ZERO,
            radius: 1.0,
        };
        tree.nodes.len()
    ];
    // Parents precede children in node order (insert_path appends), so one forward
    // pass sees every parent's circle before its children need it.
    for n in 0..tree.nodes.len() {
        let parent = circles[n];
        let scales = child_scales(tree, n);
        for (slot, &child) in tree.nodes[n].children.iter().enumerate() {
            let Some(c) = child else { continue };
            let shrink = CIRCLE_SHRINK * scales[slot];
            let step = CIRCLE_STEP.min(1.0 - shrink);
            circles[c] = Circle {
                center: parent.center + dir_vec(DIRS[slot]) * (parent.radius * step),
                radius: (parent.radius * shrink).max(MIN_RADIUS),
            };
        }
    }
    circles
}

// ---------------------------------------------------------------------------
// CPU canvas.
// ---------------------------------------------------------------------------

const CANVAS_PX: usize = 800;

/// Width of the backdrop's edge-to-transparent fade band (rl#381), in canvas px.
const BACKDROP_FADE_PX: f64 = 110.0;

/// An inclusive canvas-clamped pixel span; empty ranges come out inverted and the
/// `for` loops simply don't run.
fn clamp_span(lo: f64, hi: f64) -> (i32, i32) {
    (
        (lo.floor() as i32).max(0),
        (hi.ceil() as i32).min(CANVAS_PX as i32 - 1),
    )
}

struct Canvas {
    px: Vec<u8>,
}

impl Canvas {
    /// Recycles `buf` (the previous frame's image data) — the ~2.5MB buffer
    /// round-trips between the Image asset and the painter instead of being
    /// reallocated every open frame.
    fn new(mut buf: Vec<u8>) -> Self {
        buf.clear();
        buf.resize(CANVAS_PX * CANVAS_PX * 4, 0);
        Self { px: buf }
    }

    /// The backdrop fill, alpha fading to fully transparent over the outer
    /// [`BACKDROP_FADE_PX`] band (rl#381): the square's hard edge against the world
    /// read as an abrupt UI seam. Smoothstep per axis, multiplied — corners round off
    /// instead of double-darkening.
    fn fill_backdrop(&mut self, c: [u8; 3], a: u8) {
        let fade: Vec<f32> = (0..CANVAS_PX)
            .map(|i| {
                let t = (i.min(CANVAS_PX - 1 - i) as f64 / BACKDROP_FADE_PX).min(1.0);
                (t * t * (3.0 - 2.0 * t)) as f32
            })
            .collect();
        for (y, row) in self.px.chunks_exact_mut(CANVAS_PX * 4).enumerate() {
            for (x, p) in row.chunks_exact_mut(4).enumerate() {
                p[..3].copy_from_slice(&c);
                p[3] = (a as f32 * fade[x] * fade[y]) as u8;
            }
        }
    }

    fn blend(&mut self, x: i32, y: i32, c: [u8; 3], a: f32) {
        if x < 0 || y < 0 || x >= CANVAS_PX as i32 || y >= CANVAS_PX as i32 || a <= 0.0 {
            return;
        }
        let i = (y as usize * CANVAS_PX + x as usize) * 4;
        let a = a.min(1.0);
        for (k, &ch) in c.iter().enumerate() {
            let dst = self.px[i + k] as f32;
            self.px[i + k] = (dst + (ch as f32 - dst) * a) as u8;
        }
        let da = self.px[i + 3] as f32 / 255.0;
        self.px[i + 3] = ((da + (1.0 - da) * a) * 255.0) as u8;
    }

    fn disc(&mut self, center: DVec2, radius: f64, c: [u8; 3], alpha: f32) {
        self.annulus(center, 0.0, radius, c, alpha);
    }

    fn ring(&mut self, center: DVec2, radius: f64, width: f64, c: [u8; 3], alpha: f32) {
        self.annulus(center, radius - width / 2.0, radius + width / 2.0, c, alpha);
    }

    /// Anti-aliased annulus, visiting only the band's own rows and, per row, only the
    /// span(s) inside the outer edge and outside the inner hole. A zoomed-out
    /// ancestor ring can be 10^5 px across — iterating its full bounding box (the
    /// naive rasterizer) is a multi-ms stall per ring, while its on-canvas band is a
    /// few thousand pixels. All the math is f64 end-to-end: at deep zoom an ancestor's
    /// center/radius reach 10^7+ px, where f32 span endpoints (cancellation of
    /// near-equal huge values) wander by whole pixels and paint phantom rows.
    fn annulus(&mut self, center: DVec2, r0: f64, r1: f64, c: [u8; 3], alpha: f32) {
        let (y0, y1) = clamp_span(center.y - r1 - 1.0, center.y + r1 + 1.0);
        for y in y0..=y1 {
            let dy = y as f64 - center.y;
            let out2 = (r1 + 1.0) * (r1 + 1.0) - dy * dy;
            if out2 <= 0.0 {
                continue;
            }
            let half_out = out2.sqrt();
            let hole2 = (r0 - 1.0).max(0.0).powi(2) - dy * dy;
            let mut spans = [(center.x - half_out, center.x + half_out), (0.0, -1.0)];
            if hole2 > 0.0 {
                let half_in = hole2.sqrt();
                spans = [
                    (center.x - half_out, center.x - half_in),
                    (center.x + half_in, center.x + half_out),
                ];
            }
            for (lo, hi) in spans {
                let (x0, x1) = clamp_span(lo, hi);
                for x in x0..=x1 {
                    let d = (DVec2::new(x as f64, y as f64) - center).length();
                    // A disc (r0 = 0) has no inner edge to anti-alias — the naive
                    // symmetric formula would dim the exact center pixel (d ≈ 0
                    // gives inner factor ~0.5), a pinhole on every landmark heart.
                    let inner = if r0 <= 0.0 {
                        1.0
                    } else {
                        (d - r0 + 0.5).clamp(0.0, 1.0)
                    };
                    let cov = (r1 + 0.5 - d).clamp(0.0, 1.0) * inner;
                    self.blend(x, y, c, cov as f32 * alpha);
                }
            }
        }
    }
}

fn dir_color(d: ChordDir) -> [u8; 3] {
    match d {
        ChordDir::Up => [79, 195, 247],
        ChordDir::Down => [255, 183, 77],
        ChordDir::Left => [186, 104, 250],
        ChordDir::Right => [129, 199, 132],
    }
}

/// A code's family color — the submission mock tints a whole wing by its first
/// press, so "the sky wing is blue" reads at every depth.
fn family_color(code: &[ChordDir]) -> [u8; 3] {
    code.first().map_or([150, 158, 172], |&d| dir_color(d))
}

// ---------------------------------------------------------------------------
// Live state + systems.
// ---------------------------------------------------------------------------

/// Room labels on screen at once; only circles big enough to read as places get
/// one, so the visible set stays small. Overflow drops labels silently.
const LABEL_POOL: usize = 12;
/// A discovered room's label appears once its circle is this big on the canvas —
/// the text-free directive's one concession, gated to the zoomed-in reading.
const LABEL_MIN_RADIUS_PX: f64 = 90.0;
/// How much of the view the focus circle's diameter fills — the rest is ancestors
/// ghosting past the edges.
const FOCUS_FILL: f64 = 0.58;

#[derive(Resource)]
pub(crate) struct ChordMapState {
    visible: f32,
    focus: Vec<ChordDir>,
    /// Animated camera: world center + view width (world units per canvas side).
    view_center: DVec2,
    view_size: f64,
    /// The just-discovered code and its VISIBLE age — drives the bloom highlight.
    /// Discovery lands on the release frame (the map is already gone — rl#381 instant
    /// close), so the bloom ages only while the map renders and the growth moment
    /// greets the player on the next open instead of playing to a closed panel.
    bloom: Option<(Vec<ChordDir>, f32)>,
}

impl Default for ChordMapState {
    fn default() -> Self {
        Self {
            visible: 0.0,
            focus: Vec::new(),
            view_center: DVec2::ZERO,
            view_size: ROOT_VIEW_SIZE,
            bloom: None,
        }
    }
}

/// The root view: the unit circle's diameter at [`FOCUS_FILL`].
const ROOT_VIEW_SIZE: f64 = 2.0 / FOCUS_FILL;

#[derive(Resource)]
struct ChordMapCanvas(Handle<Image>);

#[derive(Component)]
struct ChordMapPanel;

#[derive(Component)]
struct ChordMapLabel;

/// One wiring for every GCR surface that shows the map (windowed + offscreen
/// evidence). `save` is the discovered-set file; `None` runs unpersisted.
pub fn install(app: &mut App, save: Option<PathBuf>) {
    app.insert_resource(DiscoveredCodes::load(save))
        .init_resource::<ChordMapState>()
        .add_systems(Startup, spawn_chord_map)
        .add_systems(Update, drive_chord_map);
}

fn spawn_chord_map(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new(
        Extent3d {
            width: CANVAS_PX as u32,
            height: CANVAS_PX as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0; CANVAS_PX * CANVAS_PX * 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    let handle = images.add(image);
    commands.insert_resource(ChordMapCanvas(handle.clone()));
    // Full-screen (the owner's 2026-08-12 amendment): a dark backdrop over the whole
    // window with the square canvas letterboxed to the short screen edge, so circles
    // stay circles on any aspect ratio.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.015, 0.02, 0.04, 0.88)),
            ChordMapPanel,
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    height: Val::Percent(100.0),
                    aspect_ratio: Some(1.0),
                    ..default()
                })
                .with_children(|frame| {
                    frame.spawn((
                        ImageNode::new(handle),
                        Node {
                            position_type: PositionType::Absolute,
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                    ));
                    // Label anchors are percent-of-frame with a centering flex
                    // wrapper, so they track the canvas at any window size with no
                    // pixel math against the laid-out node.
                    for _ in 0..LABEL_POOL {
                        frame
                            .spawn((
                                Node {
                                    position_type: PositionType::Absolute,
                                    width: Val::Percent(40.0),
                                    justify_content: JustifyContent::Center,
                                    display: Display::None,
                                    ..default()
                                },
                                ChordMapLabel,
                            ))
                            .with_children(|wrap| {
                                wrap.spawn((
                                    Text::new(""),
                                    TextFont {
                                        font_size: FontSize::Px(16.0),
                                        ..default()
                                    },
                                    TextColor(Color::srgba(0.92, 0.92, 0.86, 1.0)),
                                ));
                            });
                    }
                });
        });
}

/// One label to place this frame: canvas px anchor (label centers under it), text.
struct LabelReq {
    at: Vec2,
    text: &'static str,
    alpha: f32,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn drive_chord_map(
    time: Res<Time>,
    chords: Res<Chords<GcrControls>>,
    mut discovered: ResMut<DiscoveredCodes>,
    mut state: ResMut<ChordMapState>,
    canvas_handle: Res<ChordMapCanvas>,
    mut images: ResMut<Assets<Image>>,
    mut panel: Query<&mut Node, (With<ChordMapPanel>, Without<ChordMapLabel>)>,
    mut labels: Query<(&mut Node, &Children), With<ChordMapLabel>>,
    mut texts: Query<(&mut Text, &mut TextColor)>,
) {
    let dt = time.delta_secs();

    // Discovery: an accepted resolution is a PLAYED code — onto the map, persisted.
    // Resolution IS the release frame, so the map is closing this same frame; the
    // bloom waits (visible-age only) for the next open.
    if let Some((code, accepted)) = chords.resolution()
        && accepted
        && discovered.insert(code)
    {
        state.bloom = Some((code.to_vec(), 0.0));
    }
    if let Some(entered) = chords.entered() {
        state.focus = entered.to_vec();
    }

    let Ok(mut panel_node) = panel.single_mut() else {
        return;
    };
    if !chords.capturing() {
        // rl#381 owner directive: the map is GONE the frame the X hold ends — no
        // linger, no fade-out, no close animation. The player is the bottleneck.
        state.visible = 0.0;
        if panel_node.display != Display::None {
            panel_node.display = Display::None;
        }
        return;
    }
    if state.visible == 0.0 {
        // Open edge (instant close pins visible to exactly 0.0 while shut): start
        // the zoom from the root pose, not wherever the last capture left off.
        state.view_center = DVec2::ZERO;
        state.view_size = ROOT_VIEW_SIZE;
    }
    state.visible += (1.0 - state.visible) * (dt * 10.0).min(1.0);
    // Age the bloom out ENTIRELY: a spent bloom left in place would keep walking
    // its ring's rows every frame at coverage 0.
    if let Some((_, age)) = &mut state.bloom {
        *age += dt;
        if *age >= 1.2 {
            state.bloom = None;
        }
    }
    panel_node.display = Display::Flex;

    let focus_code = state.focus.clone();
    let tree = MapTree::build(&discovered.visible(), &focus_code);
    let focus = tree
        .index_of(&focus_code)
        .expect("the entered path was just inserted into the tree");
    let circles = layout_circles(&tree);

    // Camera glide toward the focus circle, zoom in log space so a multi-level dive
    // moves at constant perceived speed.
    let target_center = circles[focus].center;
    let target_size = 2.0 * circles[focus].radius / FOCUS_FILL;
    let ease = 1.0 - (-dt as f64 * 9.0).exp();
    let center_step = (target_center - state.view_center) * ease;
    state.view_center += center_step;
    state.view_size =
        (state.view_size.ln() + (target_size.ln() - state.view_size.ln()) * ease).exp();
    let view_center = state.view_center;
    let scale = CANVAS_PX as f64 / state.view_size;
    // World → canvas px, f64 throughout — the painter is f64 too, because a deep
    // zoom puts ancestor centers/radii at 10^7+ px where f32 loses whole pixels.
    let to_px =
        |p: DVec2| -> DVec2 { (p - view_center) * scale + DVec2::splat(CANVAS_PX as f64 / 2.0) };

    let Some(mut image) = images.get_mut(&canvas_handle.0) else {
        return;
    };
    let mut canvas = Canvas::new(image.data.take().unwrap_or_default());
    canvas.fill_backdrop([6, 8, 13], 242);
    let alpha = state.visible;
    let mut label_reqs: Vec<LabelReq> = Vec::new();

    // Shallow-first (node order): deep circles draw over ancestor ghosts.
    for (i, node) in tree.nodes.iter().enumerate() {
        let c = circles[i];
        let px_r = c.radius * scale;
        let at = to_px(c.center);
        let off_canvas = at.x + px_r + 1.0 < 0.0
            || at.y + px_r + 1.0 < 0.0
            || at.x - px_r - 1.0 > CANVAS_PX as f64
            || at.y - px_r - 1.0 > CANVAS_PX as f64;
        if px_r < 2.5 || off_canvas {
            continue;
        }
        let color = family_color(&node.code);
        // Huge ancestor rings ghost past the edges instead of shouting.
        let ghost: f32 = if px_r > CANVAS_PX as f64 * 0.6 {
            0.3
        } else {
            1.0
        };
        let stroke = (px_r / 40.0).clamp(1.5, 4.0);
        if node.provisional {
            canvas.ring(at, px_r, stroke, [150, 155, 165], 0.45 * alpha);
        } else {
            canvas.ring(at, px_r, stroke, color, 0.75 * ghost * alpha);
        }
        // A discovered room: a filled heart, the submission's landmark mark. Skipped
        // once the heart outgrows the whole canvas (you're inside the room — it
        // would read as a near-invisible wash, and a canvas-covering disc is the one
        // rasterizer call whose row spans are all full-width).
        if node.command.is_some() && !node.provisional && px_r < CANVAS_PX as f64 {
            let rr = (px_r * 0.28).min(px_r - 1.0);
            canvas.disc(at, rr, color, 0.55 * ghost * alpha);
            canvas.disc(at, rr * 0.45, [235, 232, 218], 0.9 * ghost * alpha);
            let anchor = Vec2::new(at.x as f32, (at.y + rr + 10.0) as f32);
            if px_r > LABEL_MIN_RADIUS_PX
                && anchor.x >= 0.0
                && anchor.x <= CANVAS_PX as f32
                && anchor.y <= CANVAS_PX as f32
                && let Some(label) = node.command
            {
                label_reqs.push(LabelReq {
                    at: anchor,
                    text: label,
                    // Ghosted circle, ghosted label — full-strength text over a
                    // nearly invisible ring would shout.
                    alpha: 0.9 * ghost * alpha,
                });
            }
        }
        if i == focus {
            canvas.ring(at, px_r + 5.0, 2.5, [240, 235, 210], alpha);
        }
        if let Some((code, age)) = &state.bloom
            && *code == node.code
        {
            let t = (age / 1.2).min(1.0);
            canvas.ring(
                at,
                px_r + 6.0 + t as f64 * 40.0,
                3.0,
                [255, 240, 180],
                (1.0 - t) * alpha,
            );
        }
        // Undiscovered directions as quiet seed rings (rl#380: empty space stays
        // quiet) — only inside a discovered circle that's big on screen, so the
        // sparse space whispers "it keeps going" without teasing content. Nominal
        // size/offset, deliberately NOT density-scaled: an empty direction has no
        // weight to speak of.
        if !node.provisional && px_r > SEED_MIN_PARENT_PX {
            for (slot, &child) in node.children.iter().enumerate() {
                if child.is_some() {
                    continue;
                }
                let sr = c.radius * CIRCLE_SHRINK * SEED_SHRINK;
                let sc = c.center + dir_vec(DIRS[slot]) * (c.radius * CIRCLE_STEP);
                canvas.ring(to_px(sc), sr * scale, 1.0, [120, 128, 140], 0.12 * alpha);
            }
        }
    }

    image.data = Some(canvas.px);

    // Labels: percent-anchored wrappers (each centers its text), sorted nearest the
    // view center first so pool overflow drops the farthest (least relevant) rooms.
    let canvas_mid = Vec2::splat(CANVAS_PX as f32 / 2.0);
    label_reqs.sort_by(|a, b| {
        a.at.distance_squared(canvas_mid)
            .total_cmp(&b.at.distance_squared(canvas_mid))
    });
    let mut pool = labels.iter_mut();
    for req in label_reqs.into_iter().take(LABEL_POOL) {
        let Some((mut node, children)) = pool.next() else {
            break;
        };
        // The wrapper is 40% wide and centers its content: anchor its center on the
        // canvas point.
        node.left = Val::Percent((req.at.x / CANVAS_PX as f32 * 100.0) - 20.0);
        node.top = Val::Percent(req.at.y / CANVAS_PX as f32 * 100.0);
        node.display = Display::Flex;
        if let Some(&child) = children.first()
            && let Ok((mut text, mut color)) = texts.get_mut(child)
        {
            if text.0 != req.text {
                text.0 = req.text.to_string();
            }
            color.0 = Color::srgba(0.92, 0.92, 0.86, req.alpha);
        }
    }
    for (mut node, _) in pool {
        if node.display != Display::None {
            node.display = Display::None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ChordDir::{Down as D, Left as L, Right as R, Up as U};

    fn set(codes: &[&[ChordDir]]) -> BTreeSet<Vec<ChordDir>> {
        codes.iter().map(|c| c.to_vec()).collect()
    }

    #[test]
    fn code_serialization_round_trips() {
        let code = vec![U, U, D, D, L, R];
        assert_eq!(code_to_str(&code), "UUDDLR");
        assert_eq!(code_from_str("UUDDLR"), Some(code));
        assert_eq!(code_from_str("UX"), None);
    }

    /// `dir_index` is `as usize`, valid only while `DIRS` lists `ChordDir` in
    /// declaration order — the tie-down for every `DIRS[dir_index(d)]` site.
    #[test]
    fn dirs_table_matches_dir_index() {
        for d in DIRS {
            assert_eq!(DIRS[dir_index(d)], d);
        }
    }

    /// A bare modifier tap (the empty code) is never a discovery: its room is the
    /// root, and a blank save line couldn't round-trip anyway.
    #[test]
    fn empty_code_is_not_a_discovery() {
        let mut codes = DiscoveredCodes::load(None);
        assert!(!codes.insert(&[]));
        assert!(codes.codes().is_empty());
    }

    #[test]
    fn discovered_set_persists_and_reloads() {
        let dir = std::env::temp_dir().join(format!("chord-map-test-{}", std::process::id()));
        let file = dir.join("codes.txt");
        let _ = std::fs::remove_file(&file);
        let mut fresh = DiscoveredCodes::load(Some(file.clone()));
        assert!(fresh.codes().is_empty(), "absent file = fresh save");
        assert!(fresh.insert(&[D, L, U]));
        assert!(!fresh.insert(&[D, L, U]), "re-playing is not a discovery");
        assert!(fresh.insert(&[L]));
        let reloaded = DiscoveredCodes::load(Some(file.clone()));
        assert_eq!(reloaded.codes(), fresh.codes());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// rl#398: the session's replicated set joins the RENDERED map but never the
    /// SAVE — remote codes are the party's map, not this player's progression.
    #[test]
    fn remote_codes_render_but_never_persist() {
        let mut codes = DiscoveredCodes::load(None);
        assert!(codes.insert(&[U, L]));
        codes.set_remote(set(&[&[D, D], &[U, L]]));
        assert_eq!(codes.visible(), set(&[&[U, L], &[D, D]]));
        assert_eq!(codes.codes(), &set(&[&[U, L]]), "save is local plays only");
        // A wholesale re-adopt replaces (a session leave/rejoin shrinks it back).
        codes.set_remote(BTreeSet::new());
        assert_eq!(codes.visible(), set(&[&[U, L]]));
    }

    /// Discovered-only (the rl#358 directive): the tree holds played codes and their
    /// prefix junctions plus the live entered path — NOTHING else from the registry,
    /// and command labels only on discovered codes.
    #[test]
    fn tree_is_discovered_plus_entered_only() {
        let tree = MapTree::build(&set(&[&[U, L], &[D, D]]), &[D, R]);
        // Nodes: root, U, UL, D, DD, DR — and no other registry code.
        assert_eq!(tree.nodes.len(), 6);
        let ul = tree.index_of(&[U, L]).unwrap();
        assert_eq!(tree.nodes[ul].command, Some("Enter plane"));
        assert!(!tree.nodes[ul].provisional);
        let dr = tree.index_of(&[D, R]).unwrap();
        assert!(
            tree.nodes[dr].provisional,
            "entered-but-unplayed is tentative"
        );
        assert_eq!(
            tree.nodes[dr].command, None,
            "an undiscovered junction shows no command"
        );
        assert!(tree.index_of(&[U, R]).is_none(), "no teaser branches");
    }

    /// The circular address scheme: children nest inside the parent circle in their
    /// d-pad direction, strictly contained, shrinking with depth — button-true by
    /// construction.
    #[test]
    fn circles_nest_in_button_directions() {
        let tree = MapTree::build(&set(&[&[U], &[D], &[L], &[R], &[D, U], &[D, U, U, U]]), &[]);
        let circles = layout_circles(&tree);
        for d in DIRS {
            let c = circles[tree.index_of(&[d]).unwrap()];
            let delta = c.center.normalize();
            assert!(delta.dot(dir_vec(d)) > 0.999, "{d:?} circle off-axis");
            assert!(
                c.center.length() + c.radius <= 1.0 + 1e-9,
                "{d:?} circle escapes its parent"
            );
        }
        let deep = circles[tree.index_of(&[D, U, U, U]).unwrap()];
        let mid = circles[tree.index_of(&[D, U]).unwrap()];
        assert!(
            (deep.center - mid.center).length() + deep.radius <= mid.radius + 1e-9,
            "a room must sit inside every ancestor circle"
        );
        assert!(deep.radius < mid.radius);
    }

    /// rl#380 density sizing: at a fork whose only known continuation is one
    /// direction, that child renders LARGER than a sibling on an evenly-split fork —
    /// and every density-boosted circle still nests inside its parent.
    #[test]
    fn sole_known_fork_child_renders_larger() {
        // Root forks evenly (U and D each carry one code); D's subtree then has a
        // sole continuation (DU) three deep.
        let tree = MapTree::build(&set(&[&[U, L], &[D, U, U, U]]), &[]);
        let circles = layout_circles(&tree);
        let even = circles[tree.index_of(&[U]).unwrap()];
        let sole = circles[tree.index_of(&[D, U]).unwrap()];
        let d = circles[tree.index_of(&[D]).unwrap()];
        // DU is D's only child → full share; U splits the root with D → half share.
        assert!(
            sole.radius / d.radius > even.radius / 1.0 + 0.1,
            "sole continuation {:.3} of parent vs even fork {:.3}",
            sole.radius / d.radius,
            even.radius
        );
        for (i, c) in circles.iter().enumerate().skip(1) {
            let parent = circles[tree.parent_of(i).unwrap()];
            assert!(
                (c.center - parent.center).length() + c.radius <= parent.radius + 1e-9,
                "{:?} escapes its parent",
                tree.nodes[i].code
            );
        }
    }

    /// Uncapped codes (rl#380): a deep address keeps a distinct, positive-radius
    /// circle strictly inside its parent — the f64 geometry doesn't collapse where
    /// f32 would (child offsets fall below f32 epsilon around depth 14).
    #[test]
    fn deep_codes_keep_distinct_circles() {
        let code: Vec<ChordDir> = (0..24).map(|i| [U, R, D, L][i % 4]).collect();
        let tree = MapTree::build(&set(&[&code]), &[]);
        let circles = layout_circles(&tree);
        for depth in 1..=24 {
            let i = tree.index_of(&code[..depth]).unwrap();
            let p = tree.parent_of(i).unwrap();
            let (c, parent) = (circles[i], circles[p]);
            assert!(c.radius > 0.0, "depth {depth} radius collapsed");
            assert!(
                (c.center - parent.center).length() > c.radius * 0.1,
                "depth {depth} lost its offset from the parent"
            );
        }
    }

    /// rl#384: at absurd depth (~1150+ zoom levels) the per-level shrink used to
    /// underflow the focus radius to 0, and `ln(0)` NaN'd the camera → canvas. The
    /// [`MIN_RADIUS`] floor keeps the camera math finite at any depth.
    #[test]
    fn extreme_depth_keeps_camera_math_finite() {
        let code: Vec<ChordDir> = (0..1300).map(|i| [U, R, D, L][i % 4]).collect();
        let tree = MapTree::build(&set(&[&code]), &[]);
        let circles = layout_circles(&tree);
        let focus = circles[tree.index_of(&code).unwrap()];
        assert!(focus.radius >= MIN_RADIUS, "radius underflowed");
        let target_size = 2.0 * focus.radius / FOCUS_FILL;
        assert!(target_size.ln().is_finite());
        assert!((CANVAS_PX as f64 / target_size).is_finite());
    }
}
