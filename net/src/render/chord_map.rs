//! The d-pad combo map (rl#358): a spatial, DISCOVERED-ONLY presentation of the chord
//! code space, shown while a code is being entered (the held-modifier capture) and for
//! a short linger after a code resolves so the growth moment is visible.
//!
//! Owner directives (rl#358, 2026-08-11) this module is built around:
//! - **Discovered-only**: the map renders codes the player has PLAYED (accepted
//!   resolutions), persisted per save file — no greyed-out future branches, no teasers.
//!   The satisfaction is watching the map grow more complex over time.
//! - **Button-true directions** (hyperbolic variants): the map re-roots on the current
//!   node every press — the four NEXT steps and the back-edge always leave the focus in
//!   their physical button directions; only older path segments bend.
//!
//! This is a sanctioned three-way experiment population (one seam, [`MapLayout`]): the
//! owner A/Bs the three layouts IN GAME — cycled live with M / right-stick-click while
//! the map is open — and the losers get deleted down to the enum afterwards. Shared
//! data model (discovered tree), shared input, three projection/paint impls.
//!
//! Rendering is a CPU-painted canvas ([`Canvas`] → `Image` → UI node) plus a pooled set
//! of UI text labels — no egui (the offscreen evidence app has none) and no gizmos (no
//! 2D camera), so the same path draws in the windowed client and in fp-screenshot
//! evidence captures.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crab_world::chord::{ChordDir, Chords};

use crate::controls::{GCR_CHORDS, GcrControls};

/// The layout under A/B — THE experiment seam. After the owner picks by feel, the
/// losing variants die and this enum collapses to the winner.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapLayout {
    /// Poincaré-disk glide, minimal dress: luminous direction-colored edges, the
    /// motion is the code.
    Hyperbolic,
    /// Nested d-pad quadrants: each press dives into the quadrant that holds the
    /// code, rooms live at their address.
    ZoomQuad,
    /// The hyperbolic atlas: same disk, dressed as a place — district tints,
    /// landmark names, pitch glyphs on edges, the entered code on a staff.
    HyperAtlas,
}

impl MapLayout {
    pub fn label(self) -> &'static str {
        match self {
            MapLayout::Hyperbolic => "hyperbolic glide",
            MapLayout::ZoomQuad => "zoom quadrants",
            MapLayout::HyperAtlas => "hyperbolic atlas",
        }
    }

    fn next(self) -> Self {
        match self {
            MapLayout::Hyperbolic => MapLayout::ZoomQuad,
            MapLayout::ZoomQuad => MapLayout::HyperAtlas,
            MapLayout::HyperAtlas => MapLayout::Hyperbolic,
        }
    }

    /// CLI id → layout (`--chord-map-layout`); an unknown id is an error naming the
    /// valid ids, same contract as `--show-controls-context` (rl#275).
    pub fn from_id(id: &str) -> Result<Self, String> {
        match id {
            "hyperbolic" => Ok(MapLayout::Hyperbolic),
            "zoomquad" => Ok(MapLayout::ZoomQuad),
            "hyperatlas" => Ok(MapLayout::HyperAtlas),
            _ => Err(format!(
                "chord-map layout {id:?} is unknown; want one of: hyperbolic, zoomquad, hyperatlas"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Discovered set — the per-save persistent record of played codes.
// ---------------------------------------------------------------------------

const DIRS: [ChordDir; 4] = [
    ChordDir::Up,
    ChordDir::Down,
    ChordDir::Left,
    ChordDir::Right,
];

fn dir_index(d: ChordDir) -> usize {
    match d {
        ChordDir::Up => 0,
        ChordDir::Down => 1,
        ChordDir::Left => 2,
        ChordDir::Right => 3,
    }
}

fn opposite(d: ChordDir) -> ChordDir {
    match d {
        ChordDir::Up => ChordDir::Down,
        ChordDir::Down => ChordDir::Up,
        ChordDir::Left => ChordDir::Right,
        ChordDir::Right => ChordDir::Left,
    }
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

fn code_from_str(s: &str) -> Option<Vec<ChordDir>> {
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
    file: Option<PathBuf>,
}

impl DiscoveredCodes {
    pub fn load(file: Option<PathBuf>) -> Self {
        let mut codes = BTreeSet::new();
        if let Some(path) = &file
            && let Ok(text) = std::fs::read_to_string(path)
        {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                match code_from_str(line) {
                    Some(code) => {
                        codes.insert(code);
                    }
                    None => warn!("chord-map save {}: bad line {line:?}", path.display()),
                }
            }
        }
        Self { codes, file }
    }

    /// Record a played code; returns whether it is NEW (the growth moment). Saves
    /// eagerly — discoveries are rare and the file is tiny.
    fn insert(&mut self, code: &[ChordDir]) -> bool {
        if !self.codes.insert(code.to_vec()) {
            return false;
        }
        if let Some(path) = &self.file {
            let body: String = self.codes.iter().map(|c| code_to_str(c) + "\n").collect();
            let write = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(path, body));
            if let Err(e) = write {
                warn!("chord-map save {} failed: {e}", path.display());
            }
        }
        true
    }

    fn codes(&self) -> &BTreeSet<Vec<ChordDir>> {
        &self.codes
    }
}

/// The windowed client's save location: `$RL_CHORD_MAP_FILE`, else
/// `$XDG_DATA_HOME|~/.local/share` + `giant-crab-rescue/discovered-codes.txt`.
/// `None` (no HOME at all) runs the map unpersisted rather than refusing to boot.
pub fn default_save_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RL_CHORD_MAP_FILE") {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("giant-crab-rescue/discovered-codes.txt"))
}

// ---------------------------------------------------------------------------
// The discovered tree — shared data model for all three layouts.
// ---------------------------------------------------------------------------

struct MapNode {
    code: Vec<ChordDir>,
    children: [Option<usize>; 4],
    parent: Option<(usize, ChordDir)>,
    /// The registered command's label, shown ONLY once discovered.
    command: Option<&'static str>,
    /// On the live entered path only — not (yet) part of the discovered map.
    provisional: bool,
}

struct MapTree {
    nodes: Vec<MapNode>,
}

impl MapTree {
    /// Every prefix of every discovered code, plus the live entered path as
    /// provisional nodes — the focus must exist on the map for re-rooting to mean
    /// anything, even while typing into undiscovered space.
    fn build(discovered: &BTreeSet<Vec<ChordDir>>, entered: &[ChordDir]) -> Self {
        let mut tree = Self {
            nodes: vec![MapNode {
                code: Vec::new(),
                children: [None; 4],
                parent: None,
                command: None,
                provisional: true,
            }],
        };
        for code in discovered {
            let leaf = tree.insert_path(code);
            tree.nodes[leaf].command = GCR_CHORDS
                .entries()
                .iter()
                .find(|e| e.code == code.as_slice())
                .map(|e| e.label);
            // The whole prefix chain is discovered structure.
            let mut n = Some(leaf);
            while let Some(i) = n {
                tree.nodes[i].provisional = false;
                n = tree.nodes[i].parent.map(|(p, _)| p);
            }
        }
        tree.insert_path(entered);
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
                        parent: Some((at, d)),
                        command: None,
                        provisional: true,
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

    /// Undirected edges at `n`: `(neighbor, nominal angle at n, nominal angle of the
    /// edge back at the neighbor)`. Nominal angles are the button-true compass
    /// directions: a child via press `d` leaves at `d`'s angle; the parent sits
    /// opposite the press that created `n`.
    fn edges(&self, n: usize) -> Vec<(usize, f32, f32)> {
        let mut out = Vec::new();
        if let Some((p, d)) = self.nodes[n].parent {
            out.push((p, dir_angle(opposite(d)), dir_angle(d)));
        }
        for d in DIRS {
            if let Some(c) = self.nodes[n].children[dir_index(d)] {
                out.push((c, dir_angle(d), dir_angle(opposite(d))));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Hyperbolic layout (Poincaré disk), re-rooted on the focus.
// ---------------------------------------------------------------------------

/// Canvas-space (y down) unit vector of a button direction.
fn dir_vec(d: ChordDir) -> Vec2 {
    match d {
        ChordDir::Up => Vec2::new(0.0, -1.0),
        ChordDir::Down => Vec2::new(0.0, 1.0),
        ChordDir::Left => Vec2::new(-1.0, 0.0),
        ChordDir::Right => Vec2::new(1.0, 0.0),
    }
}

fn dir_angle(d: ChordDir) -> f32 {
    let v = dir_vec(d);
    v.y.atan2(v.x)
}

fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

// Complex arithmetic on Vec2 (x + iy) — enough Möbius machinery for the disk.
fn cmul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

fn conj(a: Vec2) -> Vec2 {
    Vec2::new(a.x, -a.y)
}

fn cdiv(a: Vec2, b: Vec2) -> Vec2 {
    cmul(a, conj(b)) / b.length_squared().max(1e-12)
}

/// The Möbius translation taking 0 → a, applied to z.
fn mobius(a: Vec2, z: Vec2) -> Vec2 {
    cdiv(a + z, Vec2::X + cmul(conj(a), z))
}

/// z as seen from a recentered on the origin: the translation taking a → 0.
fn recenter(a: Vec2, z: Vec2) -> Vec2 {
    cdiv(z - a, Vec2::X - cmul(conj(a), z))
}

/// Euclidean offset of a one-press step when the source node sits at the origin.
const HYPER_STEP: f32 = 0.48;
/// Angular compression for edges past the focus ring — keeps a subtree inside its
/// outward wedge so re-rooted branches can't fold back over each other. The focus's
/// OWN edges are exact by construction (the button-true requirement).
const HYPER_COMPRESS: f32 = 0.55;

/// Positions (disk coords, |z| < 1) for every tree node, re-rooted so the focus sits
/// at the origin with every one of its exits leaving in its physical button
/// direction. When a child and the back-edge share a direction (a code like `UUD`
/// puts the parent AND the `D` child of `UU` both below it), both stay collinear —
/// direction stays true — and the parent leg is pushed farther out to keep the two
/// nodes distinct.
fn layout_hyperbolic(tree: &MapTree, focus: usize) -> Vec<Vec2> {
    let mut pos = vec![Vec2::ZERO; tree.nodes.len()];
    let mut seen = vec![false; tree.nodes.len()];
    seen[focus] = true;
    // (node, heading: outward screen angle, back-edge nominal angle at node)
    let mut queue = std::collections::VecDeque::new();
    let focus_edges = tree.edges(focus);
    for &(nb, ang, back) in &focus_edges {
        let is_parent = tree.nodes[focus].parent.is_some_and(|(p, _)| p == nb);
        let collides = is_parent
            && focus_edges
                .iter()
                .any(|&(o, oa, _)| o != nb && (oa - ang).abs() < 1e-3);
        let step = if collides {
            HYPER_STEP * 1.85
        } else {
            HYPER_STEP
        };
        let z = step * Vec2::new(ang.cos(), ang.sin());
        pos[nb] = z;
        seen[nb] = true;
        let heading = {
            let b = recenter(z, Vec2::ZERO);
            b.y.atan2(b.x) + std::f32::consts::PI
        };
        queue.push_back((nb, heading, back));
    }
    while let Some((n, heading, back_nominal)) = queue.pop_front() {
        let edges = tree.edges(n);
        for &(nb, ang, back) in &edges {
            if seen[nb] {
                continue;
            }
            let rel = wrap_angle(ang - (back_nominal + std::f32::consts::PI));
            let screen = heading + rel * HYPER_COMPRESS;
            let is_parent = tree.nodes[n].parent.is_some_and(|(p, _)| p == nb);
            let collides = is_parent
                && edges
                    .iter()
                    .any(|&(o, oa, _)| o != nb && !seen[o] && (oa - ang).abs() < 1e-3);
            let step = if collides {
                HYPER_STEP * 1.85
            } else {
                HYPER_STEP
            };
            let z = mobius(pos[n], step * Vec2::new(screen.cos(), screen.sin()));
            pos[nb] = z;
            seen[nb] = true;
            let heading_nb = {
                let b = recenter(z, pos[n]);
                b.y.atan2(b.x) + std::f32::consts::PI
            };
            queue.push_back((nb, heading_nb, back));
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// ZoomQuad layout — nested d-pad quadrants; the address IS the position.
// ---------------------------------------------------------------------------

/// Child scale per depth; the four children sit in a d-pad diamond inside the parent.
const QUAD_SCALE: f32 = 0.34;
/// Child center offset from the parent center, as a fraction of the parent size.
const QUAD_OFFSET: f32 = 0.30;
/// How much of the view the focus square fills — the rest is the parent ghosting
/// past the edges.
const QUAD_FOCUS_FILL: f32 = 0.58;

/// A code's room in unit space (root = the unit square, y down so Up is up).
fn quad_rect(code: &[ChordDir]) -> Rect {
    let mut center = Vec2::splat(0.5);
    let mut size = 1.0;
    for &d in code {
        center += dir_vec(d) * (size * QUAD_OFFSET);
        size *= QUAD_SCALE;
    }
    Rect::from_center_size(center, Vec2::splat(size))
}

fn quad_view(focus: &[ChordDir]) -> Rect {
    let r = quad_rect(focus);
    Rect::from_center_size(r.center(), r.size() / QUAD_FOCUS_FILL)
}

// ---------------------------------------------------------------------------
// CPU canvas — the one painter both disk layouts and the quadrants draw with.
// ---------------------------------------------------------------------------

pub(crate) const CANVAS_PX: usize = 460;
const DISK_SCALE: f32 = 0.46 * CANVAS_PX as f32;
const CANVAS_CENTER: Vec2 = Vec2::new(CANVAS_PX as f32 / 2.0, CANVAS_PX as f32 / 2.0);

struct Canvas {
    px: Vec<u8>,
}

impl Canvas {
    fn new() -> Self {
        Self {
            px: vec![0; CANVAS_PX * CANVAS_PX * 4],
        }
    }

    fn fill(&mut self, c: [u8; 4]) {
        for p in self.px.chunks_exact_mut(4) {
            p.copy_from_slice(&c);
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

    /// Anti-aliased thick segment via distance-to-segment over its bounding box.
    fn line(&mut self, a: Vec2, b: Vec2, width: f32, c: [u8; 3], alpha: f32) {
        let r = width / 2.0 + 1.0;
        let (x0, x1) = (a.x.min(b.x) - r, a.x.max(b.x) + r);
        let (y0, y1) = (a.y.min(b.y) - r, a.y.max(b.y) + r);
        let ab = b - a;
        let len2 = ab.length_squared().max(1e-6);
        for y in y0.floor() as i32..=y1.ceil() as i32 {
            for x in x0.floor() as i32..=x1.ceil() as i32 {
                let p = Vec2::new(x as f32, y as f32);
                let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
                let d = (p - (a + ab * t)).length();
                let cov = (width / 2.0 + 0.5 - d).clamp(0.0, 1.0);
                self.blend(x, y, c, cov * alpha);
            }
        }
    }

    fn disc(&mut self, center: Vec2, radius: f32, c: [u8; 3], alpha: f32) {
        self.annulus(center, 0.0, radius, c, alpha);
    }

    fn ring(&mut self, center: Vec2, radius: f32, width: f32, c: [u8; 3], alpha: f32) {
        self.annulus(center, radius - width / 2.0, radius + width / 2.0, c, alpha);
    }

    fn annulus(&mut self, center: Vec2, r0: f32, r1: f32, c: [u8; 3], alpha: f32) {
        let r = r1 + 1.0;
        for y in (center.y - r).floor() as i32..=(center.y + r).ceil() as i32 {
            for x in (center.x - r).floor() as i32..=(center.x + r).ceil() as i32 {
                let d = (Vec2::new(x as f32, y as f32) - center).length();
                let cov = (r1 + 0.5 - d).clamp(0.0, 1.0) * (d - r0 + 0.5).clamp(0.0, 1.0);
                self.blend(x, y, c, cov * alpha);
            }
        }
    }

    /// A soft radial splash — the atlas district tints.
    fn soft_disc(&mut self, center: Vec2, radius: f32, c: [u8; 3], alpha: f32) {
        for y in (center.y - radius).floor() as i32..=(center.y + radius).ceil() as i32 {
            for x in (center.x - radius).floor() as i32..=(center.x + radius).ceil() as i32 {
                let d = (Vec2::new(x as f32, y as f32) - center).length() / radius;
                if d < 1.0 {
                    let t = 1.0 - d * d;
                    self.blend(x, y, c, alpha * t * t);
                }
            }
        }
    }

    fn rect_stroke(&mut self, r: Rect, width: f32, c: [u8; 3], alpha: f32) {
        let (a, b) = (r.min, r.max);
        self.line(a, Vec2::new(b.x, a.y), width, c, alpha);
        self.line(Vec2::new(b.x, a.y), b, width, c, alpha);
        self.line(b, Vec2::new(a.x, b.y), width, c, alpha);
        self.line(Vec2::new(a.x, b.y), a, width, c, alpha);
    }

    fn rect_fill(&mut self, r: Rect, c: [u8; 3], alpha: f32) {
        for y in r.min.y.max(0.0) as i32..=(r.max.y.min(CANVAS_PX as f32 - 1.0)) as i32 {
            for x in r.min.x.max(0.0) as i32..=(r.max.x.min(CANVAS_PX as f32 - 1.0)) as i32 {
                self.blend(x, y, c, alpha);
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

/// rl#359: each direction is a note; rung index on the 4-rung pitch glyph, L lowest.
fn pitch(d: ChordDir) -> usize {
    match d {
        ChordDir::Left => 0,
        ChordDir::Down => 1,
        ChordDir::Right => 2,
        ChordDir::Up => 3,
    }
}

/// The hyperatlas's named districts — shown once ANY code inside is discovered.
const DISTRICTS: [(&[ChordDir], &str, [u8; 3]); 5] = [
    (&[ChordDir::Up], "Sky Harbor", [46, 96, 128]),
    (
        &[ChordDir::Up, ChordDir::Up],
        "Render Observatory",
        [70, 80, 140],
    ),
    (
        &[ChordDir::Down, ChordDir::Up],
        "Night-Bloom Garden",
        [128, 62, 118],
    ),
    (
        &[ChordDir::Down, ChordDir::Left],
        "Loam Quarter",
        [120, 88, 44],
    ),
    (
        &[ChordDir::Down, ChordDir::Right],
        "The Watershed",
        [34, 108, 98],
    ),
];

// ---------------------------------------------------------------------------
// Live state + systems.
// ---------------------------------------------------------------------------

/// Seconds the map lingers after an ACCEPTED code — long enough to watch the new
/// node bloom in.
const LINGER_ACCEPTED_S: f32 = 2.4;
const LINGER_UNKNOWN_S: f32 = 0.8;
const LABEL_POOL: usize = 34;

#[derive(Resource)]
pub struct ChordMapState {
    pub layout: MapLayout,
    visible: f32,
    linger: f32,
    focus: Vec<ChordDir>,
    /// Animated disk positions, keyed by code — both hyperbolic variants glide
    /// through it (the interpolation IS the Möbius-recenter animation).
    hyper_pos: HashMap<Vec<ChordDir>, Vec2>,
    /// Animated zoomquad camera, unit space.
    view: Rect,
    /// The just-discovered code and its age — drives the bloom highlight.
    bloom: Option<(Vec<ChordDir>, f32)>,
    was_open: bool,
}

impl ChordMapState {
    pub fn new(layout: MapLayout) -> Self {
        Self {
            layout,
            visible: 0.0,
            linger: 0.0,
            focus: Vec::new(),
            hyper_pos: HashMap::new(),
            view: Rect::from_center_size(Vec2::splat(0.5), Vec2::ONE),
            bloom: None,
            was_open: false,
        }
    }
}

impl Default for ChordMapState {
    fn default() -> Self {
        Self::new(MapLayout::Hyperbolic)
    }
}

#[derive(Resource)]
struct ChordMapCanvas(Handle<Image>);

#[derive(Component)]
struct ChordMapPanel;

#[derive(Component)]
struct ChordMapHeader;

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
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(24.0),
                top: Val::Px(96.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(6.0),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.78)),
            ChordMapPanel,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new(""),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.87, 0.8)),
                ChordMapHeader,
            ));
            panel
                .spawn(Node {
                    width: Val::Px(CANVAS_PX as f32),
                    height: Val::Px(CANVAS_PX as f32),
                    ..default()
                })
                .with_children(|frame| {
                    frame.spawn((
                        ImageNode::new(handle),
                        Node {
                            position_type: PositionType::Absolute,
                            width: Val::Px(CANVAS_PX as f32),
                            height: Val::Px(CANVAS_PX as f32),
                            ..default()
                        },
                    ));
                    for _ in 0..LABEL_POOL {
                        frame.spawn((
                            Text::new(""),
                            TextFont {
                                font_size: 13.0,
                                ..default()
                            },
                            TextColor(Color::srgba(0.9, 0.9, 0.85, 1.0)),
                            Node {
                                position_type: PositionType::Absolute,
                                display: Display::None,
                                ..default()
                            },
                            ChordMapLabel,
                        ));
                    }
                });
        });
}

/// One label to place this frame: canvas px anchor, text, srgba color.
struct LabelReq {
    at: Vec2,
    text: String,
    color: Color,
    centered: bool,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn drive_chord_map(
    time: Res<Time>,
    chords: Res<Chords<GcrControls>>,
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    mut discovered: ResMut<DiscoveredCodes>,
    mut state: ResMut<ChordMapState>,
    canvas_handle: Res<ChordMapCanvas>,
    mut images: ResMut<Assets<Image>>,
    mut panel: Query<&mut Node, (With<ChordMapPanel>, Without<ChordMapLabel>)>,
    mut header: Query<&mut Text, (With<ChordMapHeader>, Without<ChordMapLabel>)>,
    mut labels: Query<(&mut Node, &mut Text, &mut TextColor), With<ChordMapLabel>>,
) {
    let dt = time.delta_secs();

    // Discovery: an accepted resolution is a PLAYED code — onto the map, persisted.
    if let Some((code, accepted)) = chords.resolution() {
        state.focus = code.to_vec();
        if accepted {
            if discovered.insert(code) {
                state.bloom = Some((code.to_vec(), 0.0));
            }
            state.linger = LINGER_ACCEPTED_S;
        } else {
            state.linger = LINGER_UNKNOWN_S;
        }
    }
    if let Some(entered) = chords.entered() {
        if !state.was_open {
            // Fresh open: start the glide from the root pose, not wherever the last
            // linger left off.
            state.hyper_pos.clear();
            state.view = Rect::from_center_size(Vec2::splat(0.5), Vec2::ONE);
            state.bloom = None;
        }
        state.focus = entered.to_vec();
        state.linger = state.linger.max(0.1);
        state.was_open = true;
    }
    state.linger = (state.linger - dt).max(0.0);
    let open = state.linger > 0.0;
    if !open {
        state.was_open = false;
    }
    state.visible += ((open as u8 as f32) - state.visible) * (dt * 10.0).min(1.0);
    if let Some((_, age)) = &mut state.bloom {
        *age += dt;
    }

    // Live layout cycling — the A/B switch, reachable without leaving the map.
    if open
        && (keys.just_pressed(KeyCode::KeyM)
            || pads
                .iter()
                .any(|gp| gp.just_pressed(GamepadButton::RightThumb)))
    {
        state.layout = state.layout.next();
    }

    let Ok(mut panel_node) = panel.single_mut() else {
        return;
    };
    if state.visible < 0.02 {
        if panel_node.display != Display::None {
            panel_node.display = Display::None;
        }
        return;
    }
    panel_node.display = Display::Flex;

    if let Ok(mut text) = header.single_mut() {
        let want = format!(
            "chord map · {}   ({} found · M / R3: next layout)",
            state.layout.label(),
            discovered.codes().len()
        );
        if text.0 != want {
            text.0 = want;
        }
    }

    let focus_code = state.focus.clone();
    let tree = MapTree::build(discovered.codes(), &focus_code);
    let focus = tree
        .index_of(&focus_code)
        .expect("the entered path was just inserted into the tree");

    let mut canvas = Canvas::new();
    canvas.fill([6, 8, 13, 235]);
    let alpha = state.visible;
    let mut label_reqs: Vec<LabelReq> = Vec::new();

    match state.layout {
        MapLayout::Hyperbolic | MapLayout::HyperAtlas => {
            let atlas = state.layout == MapLayout::HyperAtlas;
            let target = layout_hyperbolic(&tree, focus);
            // Glide: every node eases toward its re-rooted position; a new node
            // grows out of its parent's current position.
            let ease = 1.0 - (-dt * 9.0).exp();
            let mut animated = vec![Vec2::ZERO; tree.nodes.len()];
            for (i, node) in tree.nodes.iter().enumerate() {
                let spawn_at = node
                    .parent
                    .and_then(|(p, _)| state.hyper_pos.get(&tree.nodes[p].code))
                    .copied()
                    .unwrap_or(target[i]);
                let cur = state.hyper_pos.get(&node.code).copied().unwrap_or(spawn_at);
                animated[i] = cur + (target[i] - cur) * ease;
            }
            state.hyper_pos = tree
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.code.clone(), animated[i]))
                .collect();
            let to_px = |z: Vec2| CANVAS_CENTER + z * DISK_SCALE;

            canvas.ring(CANVAS_CENTER, DISK_SCALE, 1.5, [46, 54, 70], 0.6 * alpha);
            if atlas {
                for (prefix, name, color) in DISTRICTS {
                    let members: Vec<Vec2> = tree
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| !n.provisional && n.code.starts_with(prefix))
                        .map(|(i, _)| to_px(animated[i]))
                        .collect();
                    // The district exists once something DEEPER than its gate is found.
                    if members.len() < 2 {
                        continue;
                    }
                    for &m in &members {
                        canvas.soft_disc(m, 46.0, color, 0.16 * alpha);
                    }
                    let centroid = members.iter().sum::<Vec2>() / members.len() as f32;
                    label_reqs.push(LabelReq {
                        at: centroid + Vec2::new(0.0, -30.0),
                        text: name.to_string(),
                        color: Color::srgba(
                            color[0] as f32 / 160.0,
                            color[1] as f32 / 160.0,
                            color[2] as f32 / 160.0,
                            0.75 * alpha,
                        ),
                        centered: true,
                    });
                }
            }

            for (i, node) in tree.nodes.iter().enumerate() {
                let Some((p, d)) = node.parent else {
                    continue;
                };
                let (a, b) = (to_px(animated[p]), to_px(animated[i]));
                let on_path = focus_code.starts_with(&node.code) || i == focus;
                let dim = if node.provisional {
                    0.35
                } else if on_path {
                    1.0
                } else {
                    0.7
                };
                let depth_fade = 1.0 / (1.0 + 0.12 * node.code.len() as f32);
                let width = if on_path { 3.0 } else { 2.0 };
                canvas.line(a, b, width, dir_color(d), dim * depth_fade * alpha);
                if atlas && !node.provisional {
                    // Pitch glyph: 4 rungs at the edge midpoint, the step's note lit.
                    let mid = (a + b) / 2.0;
                    for rung in 0..4 {
                        let at = mid + Vec2::new(0.0, 6.0 - 4.0 * rung as f32);
                        let lit = rung == pitch(d);
                        canvas.disc(
                            at,
                            if lit { 2.0 } else { 1.2 },
                            if lit { dir_color(d) } else { [120, 125, 135] },
                            (if lit { 0.9 } else { 0.4 }) * alpha,
                        );
                    }
                }
            }
            for (i, node) in tree.nodes.iter().enumerate() {
                let at = to_px(animated[i]);
                if node.provisional {
                    canvas.ring(at, 4.0, 1.5, [150, 155, 165], 0.6 * alpha);
                } else if node.command.is_some() {
                    let r = if atlas { 6.5 } else { 5.5 };
                    canvas.disc(at, r, [235, 232, 218], 0.95 * alpha);
                } else {
                    canvas.disc(at, 3.0, [150, 160, 175], 0.8 * alpha);
                }
                if let Some(label) = node.command
                    && animated[i].length() < 0.94
                {
                    label_reqs.push(LabelReq {
                        at: at + Vec2::new(9.0, -8.0),
                        text: label.to_string(),
                        color: Color::srgba(0.9, 0.9, 0.85, 0.9 * alpha),
                        centered: false,
                    });
                }
            }
            canvas.ring(to_px(animated[focus]), 11.0, 2.0, [240, 235, 210], alpha);
            if let Some((code, age)) = &state.bloom
                && let Some(i) = tree.index_of(code)
            {
                let t = (age / 1.2).min(1.0);
                canvas.ring(
                    to_px(animated[i]),
                    12.0 + t * 30.0,
                    2.5,
                    [255, 240, 180],
                    (1.0 - t) * alpha,
                );
            }
            if atlas {
                // The entered code as a melody on a tiny staff, bottom-left.
                let base = Vec2::new(20.0, CANVAS_PX as f32 - 16.0);
                canvas.line(
                    base,
                    base + Vec2::new(14.0 * 8.0, 0.0),
                    1.0,
                    [70, 76, 90],
                    0.7 * alpha,
                );
                for (i, &d) in focus_code.iter().enumerate() {
                    let at = base + Vec2::new(6.0 + 14.0 * i as f32, -6.0 - 7.0 * pitch(d) as f32);
                    canvas.disc(at, 3.0, dir_color(d), 0.9 * alpha);
                }
            }
        }
        MapLayout::ZoomQuad => {
            let target = quad_view(&focus_code);
            let ease = 1.0 - (-dt * 9.0).exp();
            let center = state.view.center() + (target.center() - state.view.center()) * ease;
            // Zoom in log space so a multi-level dive glides at constant speed.
            let size = state.view.size().x.max(1e-6).ln()
                + (target.size().x.max(1e-6).ln() - state.view.size().x.max(1e-6).ln()) * ease;
            state.view = Rect::from_center_size(center, Vec2::splat(size.exp()));
            let view = state.view;
            let to_px = |p: Vec2| (p - view.min) / view.size().x.max(1e-9) * CANVAS_PX as f32;

            for (i, node) in tree.nodes.iter().enumerate() {
                let r = quad_rect(&node.code);
                let px = Rect {
                    min: to_px(r.min),
                    max: to_px(r.max),
                };
                let w = px.size().x;
                if w < 5.0
                    || px.max.x < 0.0
                    || px.max.y < 0.0
                    || px.min.x > CANVAS_PX as f32
                    || px.min.y > CANVAS_PX as f32
                {
                    continue;
                }
                let last = node.code.last().copied();
                let tint = last.map_or([90, 96, 110], dir_color);
                if node.provisional {
                    canvas.rect_stroke(px, 1.5, [150, 155, 165], 0.45 * alpha);
                } else if node.command.is_some() {
                    canvas.rect_fill(px, tint, 0.10 * alpha);
                    canvas.rect_stroke(px, 2.0, tint, 0.85 * alpha);
                } else {
                    canvas.rect_stroke(px, 1.5, tint, 0.5 * alpha);
                }
                if i == focus {
                    canvas.rect_stroke(px.inflate(3.0), 2.5, [240, 235, 210], alpha);
                }
                if let Some((code, age)) = &state.bloom
                    && *code == node.code
                {
                    let t = (age / 1.2).min(1.0);
                    canvas.rect_stroke(
                        px.inflate(3.0 + t * 22.0),
                        2.5,
                        [255, 240, 180],
                        (1.0 - t) * alpha,
                    );
                }
                if let Some(label) = node.command
                    && w > 84.0
                {
                    label_reqs.push(LabelReq {
                        at: px.center(),
                        text: label.to_string(),
                        color: Color::srgba(0.92, 0.92, 0.86, 0.9 * alpha),
                        centered: true,
                    });
                }
            }
        }
    }

    if let Some(image) = images.get_mut(&canvas_handle.0) {
        image.data = Some(canvas.px);
    }

    // De-overlap: a sparse map crowds labels around the hub — nudge later labels
    // down until nothing horizontally-close sits within a line height.
    label_reqs.sort_by(|a, b| a.at.y.total_cmp(&b.at.y));
    for i in 1..label_reqs.len() {
        for j in 0..i {
            let dx = (label_reqs[i].at.x - label_reqs[j].at.x).abs();
            let dy = label_reqs[i].at.y - label_reqs[j].at.y;
            if dx < 110.0 && dy.abs() < 14.0 {
                label_reqs[i].at.y = label_reqs[j].at.y + 14.0;
            }
        }
    }

    let mut pool = labels.iter_mut();
    for req in label_reqs.into_iter().take(LABEL_POOL) {
        let Some((mut node, mut text, mut color)) = pool.next() else {
            break;
        };
        // No measured text size pre-layout; approximate centering by glyph count.
        let offset = if req.centered {
            Vec2::new(req.text.len() as f32 * 3.4, 8.0)
        } else {
            Vec2::ZERO
        };
        node.left = Val::Px(req.at.x - offset.x);
        node.top = Val::Px(req.at.y - offset.y);
        node.display = Display::Flex;
        if text.0 != req.text {
            text.0 = req.text;
        }
        color.0 = req.color;
    }
    for (mut node, _, _) in pool {
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

    /// The button-true requirement (rl#358 owner directive): every exit of the FOCUS
    /// leaves in exactly its physical button direction.
    #[test]
    fn hyperbolic_focus_exits_match_button_directions() {
        let tree = MapTree::build(&set(&[&[U, U, U], &[U, U, D], &[U, L], &[D, D]]), &[U, U]);
        let focus = tree.index_of(&[U, U]).unwrap();
        let pos = layout_hyperbolic(&tree, focus);
        assert!(pos[focus].length() < 1e-6, "focus sits at the center");
        for (child_code, d) in [(vec![U, U, U], U), (vec![U, U, D], D)] {
            let c = tree.index_of(&child_code).unwrap();
            let v = pos[c].normalize();
            let want = dir_vec(d);
            assert!(
                v.dot(want) > 0.999,
                "child {child_code:?} leaves at {v:?}, want {want:?}"
            );
        }
        // Back-edge reads true too: focus UU was entered by pressing U from U, so U
        // (the parent) sits in the Down direction.
        let parent = tree.index_of(&[U]).unwrap();
        assert!(pos[parent].normalize().dot(dir_vec(D)) > 0.999);
    }

    /// When a child and the back-edge share a direction (UU's parent is down AND its
    /// D child is down), both stay collinear — direction never lies — with the
    /// parent pushed farther out so the nodes stay distinct.
    #[test]
    fn hyperbolic_backedge_collision_stays_true_but_separates() {
        let tree = MapTree::build(&set(&[&[U, U, D]]), &[U, U]);
        let focus = tree.index_of(&[U, U]).unwrap();
        let pos = layout_hyperbolic(&tree, focus);
        let child = tree.index_of(&[U, U, D]).unwrap();
        let parent = tree.index_of(&[U]).unwrap();
        assert!(pos[child].normalize().dot(dir_vec(D)) > 0.999);
        assert!(pos[parent].normalize().dot(dir_vec(D)) > 0.999);
        assert!(
            pos[parent].length() > pos[child].length() + 0.1,
            "parent {} vs child {}",
            pos[parent].length(),
            pos[child].length()
        );
    }

    /// Every placed node stays inside the open disk (Poincaré positions are only
    /// meaningful there), across a deep discovered tree.
    #[test]
    fn hyperbolic_positions_stay_in_the_disk() {
        let codes: Vec<Vec<ChordDir>> = GCR_CHORDS
            .entries()
            .iter()
            .map(|e| e.code.to_vec())
            .collect();
        let all: BTreeSet<_> = codes.into_iter().collect();
        for focus_code in [vec![], vec![D], vec![D, U, U], vec![U, U, D, D, L, R]] {
            let tree = MapTree::build(&all, &focus_code);
            let focus = tree.index_of(&focus_code).unwrap();
            let pos = layout_hyperbolic(&tree, focus);
            for (i, z) in pos.iter().enumerate() {
                // Strictly inside the open disk — far nodes crowd the rim
                // exponentially (that's the projection working), so the bound is
                // the model's, not a pixel margin.
                assert!(
                    z.length() < 1.0,
                    "node {:?} at {z:?} escaped the disk (focus {focus_code:?})",
                    tree.nodes[i].code
                );
            }
        }
    }

    /// Distinct codes land on distinct places — the re-rooted layout must not fold
    /// two nodes onto one point (the failure the naive globally-true layout has).
    #[test]
    fn hyperbolic_positions_are_distinct() {
        let all: BTreeSet<_> = GCR_CHORDS
            .entries()
            .iter()
            .map(|e| e.code.to_vec())
            .collect();
        let tree = MapTree::build(&all, &[]);
        let pos = layout_hyperbolic(&tree, 0);
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                assert!(
                    (pos[i] - pos[j]).length() > 1e-3,
                    "{:?} and {:?} collide at {:?}",
                    tree.nodes[i].code,
                    tree.nodes[j].code,
                    pos[i]
                );
            }
        }
    }

    /// The quadrant address scheme: children sit in their d-pad direction inside the
    /// parent, nested strictly (a room is contained by every ancestor).
    #[test]
    fn quad_rects_nest_in_button_directions() {
        for d in DIRS {
            let child = quad_rect(&[d]);
            let root = quad_rect(&[]);
            let delta = (child.center() - root.center()).normalize();
            assert!(delta.dot(dir_vec(d)) > 0.999, "{d:?} room off-axis");
            assert!(root.contains(child.min) && root.contains(child.max));
        }
        let deep = quad_rect(&[D, U, U, U]);
        let mid = quad_rect(&[D, U]);
        assert!(mid.contains(deep.min) && mid.contains(deep.max));
        assert!(deep.size().x < mid.size().x);
    }

    #[test]
    fn layout_ids_round_trip_and_cycle_covers_all() {
        for (id, l) in [
            ("hyperbolic", MapLayout::Hyperbolic),
            ("zoomquad", MapLayout::ZoomQuad),
            ("hyperatlas", MapLayout::HyperAtlas),
        ] {
            assert_eq!(MapLayout::from_id(id), Ok(l));
        }
        assert!(MapLayout::from_id("nope").is_err());
        let mut l = MapLayout::Hyperbolic;
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(l);
            l = l.next();
        }
        assert_eq!(l, MapLayout::Hyperbolic);
        assert_eq!(seen.len(), 3);
    }
}
