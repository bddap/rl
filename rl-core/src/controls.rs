//! Reusable controls + hold-to-reveal-overlay framework, generic over an app's action
//! set. One executable = one [`ControlScheme`] (its action enum, input vocabulary, glyph
//! art, and control map); the framework turns that into the on-screen control legend and
//! the polished hold-to-reveal overlay. GCR ([`crate::net::controls`]) and the demo
//! ([`crate::play`]) are disjoint apps with disjoint verbs — each brings its own scheme;
//! the overlay code below is shared.
//!
//! The single-source guarantee, PER APP: a scheme's [`ControlScheme::map`] is the one
//! table the framework reads. The legend glyphs derive from it via [`ControlScheme`]'s
//! `*_glyph` resolvers, so the displayed bindings can't drift from the table. An app whose
//! live input ALSO reads the map (GCR, via the [`ControlInput`] glue + its own
//! `key_code_for`) gets a full round-trip — rebind in the table, both the poll and the
//! legend move together. An app whose input is analog/multi-key and read directly (the
//! demo) uses the map as the single source of its LEGEND, at the granularity that reads
//! well.
//!
//! Two layers, split by the `render` feature — exactly like the rest of the crate:
//! - The **pure core** ([`Device`], [`Glyph`], [`LegendLine`], [`ControlEntry`], the
//!   [`ControlScheme`] trait, [`legend`]/[`reveal_glyph`], [`assert_map_well_formed`]) has
//!   NO Bevy dependency, so it compiles and unit-tests in the no-feature build.
//! - The **Bevy glue** (the `#[cfg(feature = "render")]` block) is the only place the
//!   typed inputs meet Bevy's input API and the only place the overlay UI lives.

use std::fmt::Debug;

/// The active input device: which column of a scheme's map the legend shows and which
/// glyph set to render. Detected from recent input by [`track_active_device`] — the last
/// device that produced input wins, so a couch player who picks up the pad sees pad glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// The desktop default; the legend starts here until a pad is touched.
    #[default]
    KeyboardMouse,
    Gamepad,
}

/// One legend glyph: a bundled icon (asset path under the Bevy asset root's `controls/`
/// dir) OR a short text label for an input with no bundled icon (e.g. `"G"`, `"Arrows"`,
/// `"LT"`). The [`Glyph::Label`] arm is what lets the framework render a control without a
/// PNG for every key — a new app's overlay works before it ships art, and every control is
/// always representable (no "missing glyph" illegal state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    /// Asset path under `controls/` of a bundled icon (e.g. the Kenney Input Prompts PNGs).
    Icon(&'static str),
    /// A short text label rendered as a keycap chip when no icon is bundled.
    Label(&'static str),
}

/// An app's control scheme: its action enum, its input vocabulary, the glyph art for that
/// vocabulary, and the control map. Implemented once per executable. The associated input
/// types ([`Key`](ControlScheme::Key)/[`Pad`](ControlScheme::Pad)/[`Mouse`](ControlScheme::Mouse))
/// make the scheme self-describing, so the legend's glyphs derive from the SAME typed map
/// the input reads — no drift.
pub trait ControlScheme: 'static + Send + Sync {
    /// The controllable verbs. The row key of [`ControlScheme::map`]; the per-app
    /// invariant test ([`assert_map_well_formed`]) proves every variant has exactly one row.
    type Action: Copy + PartialEq + Debug;
    /// The app's keyboard vocabulary (a closed enum: only the keys it binds), so the map
    /// and glyph table are exhaustive and a typo can't name a nonexistent key.
    type Key: Copy + PartialEq;
    /// The app's gamepad vocabulary.
    type Pad: Copy + PartialEq;
    /// The app's mouse vocabulary (motion / buttons / wheel).
    type Mouse: Copy + PartialEq;

    /// THE control map: one row per action, in legend display order.
    fn map() -> &'static [ControlEntry<Self>];

    /// The action whose HOLD reveals the overlay (and whose glyph the corner hint shows).
    fn reveal_action() -> Self::Action;

    /// Glyph art for the input vocabulary. Total over each enum, so a binding can't name an
    /// input with no glyph (the make-illegal-states-unrepresentable half of "no drift").
    fn key_glyph(key: Self::Key) -> Glyph;
    fn pad_glyph(pad: Self::Pad) -> Glyph;
    fn mouse_glyph(mouse: Self::Mouse) -> Glyph;
}

/// How an [`Action`](ControlScheme::Action) is triggered on keyboard/mouse: ordered key +
/// mouse input lists (each list may be empty), plus a `hold` flag. Empty lists mean "no
/// keyboard binding on this device" — a legitimate, representable state (pad-only demo
/// controls use it; the legend then omits the row from the keyboard column).
pub struct KbBinding<S: ControlScheme + ?Sized> {
    /// Keys then mouse inputs, in glyph-display order.
    pub keys: &'static [S::Key],
    pub mouse: &'static [S::Mouse],
    /// A sustained hold (reveal-controls, GCR's quit) rather than a tap — the legend prefixes
    /// "Hold". Hold-ness is data per device, not a special case in [`legend`].
    pub hold: bool,
}

/// How an action is triggered on a gamepad: an ordered button list (may be empty) + a
/// `hold` flag.
pub struct PadBinding<S: ControlScheme + ?Sized> {
    /// Pad controls in glyph-display order.
    pub buttons: &'static [S::Pad],
    /// A press-and-hold rather than a tap (see [`KbBinding::hold`]).
    pub hold: bool,
}

impl<S: ControlScheme + ?Sized> KbBinding<S> {
    /// An empty keyboard binding — the action isn't bound on keyboard/mouse.
    pub const NONE: Self = Self {
        keys: &[],
        mouse: &[],
        hold: false,
    };
    /// A tap binding from key + mouse lists.
    pub const fn new(keys: &'static [S::Key], mouse: &'static [S::Mouse]) -> Self {
        Self {
            keys,
            mouse,
            hold: false,
        }
    }
    /// A hold binding (the legend prefixes "Hold").
    pub const fn hold(keys: &'static [S::Key], mouse: &'static [S::Mouse]) -> Self {
        Self {
            keys,
            mouse,
            hold: true,
        }
    }
}

impl<S: ControlScheme + ?Sized> PadBinding<S> {
    /// A tap binding from a button list.
    pub const fn new(buttons: &'static [S::Pad]) -> Self {
        Self {
            buttons,
            hold: false,
        }
    }
    /// A hold binding.
    pub const fn hold(buttons: &'static [S::Pad]) -> Self {
        Self {
            buttons,
            hold: true,
        }
    }
}

/// One row of a scheme's control map: an action, its legend label, and its binding on each
/// device.
pub struct ControlEntry<S: ControlScheme + ?Sized> {
    pub action: S::Action,
    /// Short human label for the legend (e.g. "Forward", "Rebuild crab").
    pub label: &'static str,
    pub keyboard: KbBinding<S>,
    pub pad: PadBinding<S>,
}

impl<S: ControlScheme + ?Sized> ControlEntry<S> {
    /// Whether this row is a hold on the given device.
    pub fn is_hold(&self, device: Device) -> bool {
        match device {
            Device::KeyboardMouse => self.keyboard.hold,
            Device::Gamepad => self.pad.hold,
        }
    }

    /// The ordered glyphs to display for this row on `device` (empty if it has no binding
    /// there). Keyboard: each key glyph then each mouse glyph; gamepad: each button glyph.
    pub fn glyphs(&self, device: Device) -> Vec<Glyph> {
        match device {
            Device::KeyboardMouse => self
                .keyboard
                .keys
                .iter()
                .map(|&k| S::key_glyph(k))
                .chain(self.keyboard.mouse.iter().map(|&m| S::mouse_glyph(m)))
                .collect(),
            Device::Gamepad => self.pad.buttons.iter().map(|&p| S::pad_glyph(p)).collect(),
        }
    }
}

/// Look up a scheme's row for `action`. Total over the scheme's actions (every action has a
/// row — proven by [`assert_map_well_formed`]), so the `Option` is just the lookup's honest
/// shape.
pub fn entry<S: ControlScheme + ?Sized>(action: S::Action) -> Option<&'static ControlEntry<S>> {
    S::map().iter().find(|e| e.action == action)
}

/// One ready-to-render legend line: the action's label, whether it's a hold, and the glyphs
/// to show in order. What the overlay turns into an icon/keycap row + text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegendLine {
    pub label: &'static str,
    /// True if this action is a hold — the overlay prefixes "Hold".
    pub hold: bool,
    pub glyphs: Vec<Glyph>,
}

/// Build the legend for `device` from a scheme's map: one [`LegendLine`] per row that is
/// BOUND on that device, in map order. Rows with no binding on the device are omitted (so
/// the keyboard column doesn't list a pad-only control). Iterating the map directly is what
/// keeps it the only enumeration source — the legend can't drift from the live bindings.
pub fn legend<S: ControlScheme + ?Sized>(device: Device) -> Vec<LegendLine> {
    S::map()
        .iter()
        .filter_map(|e| {
            let glyphs = e.glyphs(device);
            if glyphs.is_empty() {
                return None;
            }
            Some(LegendLine {
                label: e.label,
                hold: e.is_hold(device),
                glyphs,
            })
        })
        .collect()
}

/// The single glyph for the reveal control on `device`, for the corner hint. The reveal
/// action binds at least one control per device (its first glyph), so this never returns
/// `None` for a well-formed scheme; the `Option` documents the invariant rather than
/// inventing a fallback.
pub fn reveal_glyph<S: ControlScheme + ?Sized>(device: Device) -> Option<Glyph> {
    entry::<S>(S::reveal_action()).and_then(|e| e.glyphs(device).first().copied())
}

/// Per-app well-formedness check, called from each scheme's own `#[test]` with its
/// exhaustive action list: every action appears EXACTLY once in the map, and every action
/// is bound on at least one device (else it'd be invisible and unusable). The compiler-side
/// half — that a new `Action` variant can't be added without a map row — is each app's
/// exhaustive `match` over its actions producing `all_actions`.
pub fn assert_map_well_formed<S: ControlScheme + ?Sized>(all_actions: &[S::Action]) {
    for &a in all_actions {
        let n = S::map().iter().filter(|e| e.action == a).count();
        assert_eq!(n, 1, "{a:?} appears {n} times in the map; want exactly 1");
        let e = entry::<S>(a).expect("just checked exactly one row exists");
        assert!(
            !e.glyphs(Device::KeyboardMouse).is_empty() || !e.glyphs(Device::Gamepad).is_empty(),
            "{a:?} is bound on no device (would be invisible/unusable)"
        );
    }
    assert_eq!(
        S::map().len(),
        all_actions.len(),
        "the map has rows for actions not in the exhaustive list (a stale/duplicate row)"
    );
    // The reveal action must resolve to a hint glyph on both devices (the corner hint).
    for device in [Device::KeyboardMouse, Device::Gamepad] {
        assert!(
            reveal_glyph::<S>(device).is_some(),
            "reveal action has no glyph on {device:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Bevy glue — the typed inputs meet Bevy's input API, and the overlay UI lives here.
// Render-only: Bevy's KeyCode/GamepadButton exist only under the `render` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "render")]
mod overlay {
    use super::*;
    use bevy::prelude::*;

    /// The Bevy-input glue a scheme needs for the overlay to poll its reveal control. Only
    /// the discrete reveal key/button is polled here, so analog inputs (sticks, mouse
    /// motion, wheel) return `None` — they're never the reveal control. An app's GAMEPLAY
    /// input reading (GCR's `gather_input`) stays in the app; this is overlay-only.
    pub trait ControlInput: ControlScheme {
        /// The Bevy `KeyCode` for a key, if it maps to one (analog/non-key inputs: `None`).
        fn key_code(key: Self::Key) -> Option<KeyCode>;
        /// The Bevy `GamepadButton` for a pad control, if it's a discrete button (sticks /
        /// D-pad read via their own axis/multi-button APIs: `None`).
        fn gamepad_button(pad: Self::Pad) -> Option<GamepadButton>;
    }

    /// The input device the player is currently using, refreshed each frame by
    /// [`track_active_device`]. Drives which legend column + hint glyph show. Pure client UI.
    #[derive(Resource, Clone, Copy, Default)]
    pub struct ActiveDevice(pub Device);

    /// Force the reveal overlay open regardless of input — for a HEADLESS screenshot, which
    /// has no live keyboard/pad to hold the reveal control. Defaults false (the windowed
    /// client stays hold-to-reveal); a screenshot app sets it true for an evidence frame.
    #[derive(Resource, Clone, Copy, Default)]
    pub struct ForceRevealControls(pub bool);

    /// Marks the always-visible corner hint ("Hold [glyph] — Controls").
    #[derive(Component)]
    pub struct ControlsHintRoot;

    /// Marks the hold-to-reveal overlay root (the dark panel). Toggled between
    /// [`Display::None`] and [`Display::Flex`] by [`update_controls_ui`].
    #[derive(Component)]
    pub struct ControlsOverlayRoot;

    /// One device's legend container inside the overlay (built once per device). Only the
    /// active device's container is shown — pre-building both avoids rebuilding child
    /// entities every time the player switches device or opens the overlay.
    #[derive(Component)]
    pub struct LegendColumn(Device);

    /// One device's reveal-glyph node in the corner hint. Like [`LegendColumn`], both are
    /// pre-built (keyboard + pad) and only the active device's is shown — so the hint
    /// handles icon AND text-keycap reveal glyphs uniformly, with no per-frame asset reload.
    #[derive(Component)]
    pub struct HintGlyphFor(Device);

    /// Pixel size of a rendered glyph icon (square). Reads cleanly at TV distance.
    const GLYPH_PX: f32 = 30.0;

    /// Gamepad stick noise floor: a resting stick reads small nonzero magnitude on real
    /// hardware, so a read below this counts as "not touching the stick". One hardware
    /// fact, two consumers — [`track_active_device`] (a resting stick mustn't flip the
    /// legend off the keyboard) and GCR's `pad_stick_axes` (a resting stick mustn't creep
    /// the avatar/view) — shared so a recalibration moves both together.
    pub(crate) const PAD_STICK_DEADZONE: f32 = 0.15;

    /// Spawn a single glyph node (icon image OR text keycap chip) as a child of `parent`.
    fn spawn_glyph(parent: &mut ChildSpawnerCommands, asset_server: &AssetServer, glyph: Glyph) {
        match glyph {
            Glyph::Icon(path) => {
                parent.spawn((
                    ImageNode::new(asset_server.load(path)),
                    Node {
                        width: Val::Px(GLYPH_PX),
                        height: Val::Px(GLYPH_PX),
                        ..default()
                    },
                ));
            }
            Glyph::Label(text) => {
                // A keycap chip: a padded light-on-dark pill so a text key reads like the
                // icon glyphs beside it.
                parent
                    .spawn((
                        Node {
                            min_width: Val::Px(GLYPH_PX),
                            height: Val::Px(GLYPH_PX),
                            padding: UiRect::horizontal(Val::Px(6.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.28, 0.28, 0.30)),
                    ))
                    .with_children(|chip| {
                        chip.spawn((
                            Text::new(text),
                            TextFont {
                                font_size: 16.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.95, 0.95, 0.95)),
                        ));
                    });
            }
        }
    }

    /// Spawn one legend column (icon/label + label rows) for `device`, as a child of the
    /// overlay. Each row comes straight from [`legend`], so the panel IS the live bindings.
    fn spawn_legend_column<S: ControlScheme>(
        parent: &mut ChildSpawnerCommands,
        asset_server: &AssetServer,
        device: Device,
        visible: bool,
    ) {
        parent
            .spawn((
                Node {
                    display: if visible {
                        Display::Flex
                    } else {
                        Display::None
                    },
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(8.0),
                    ..default()
                },
                LegendColumn(device),
            ))
            .with_children(|col| {
                for line in legend::<S>(device) {
                    col.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|row| {
                        for glyph in line.glyphs {
                            spawn_glyph(row, asset_server, glyph);
                        }
                        let label = if line.hold {
                            format!("Hold {}", line.label)
                        } else {
                            line.label.to_string()
                        };
                        row.spawn((
                            Text::new(label),
                            TextFont {
                                font_size: 20.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.95, 0.95, 0.95)),
                        ));
                    });
                }
            });
    }

    /// Spawn the controls UI for scheme `S`: the always-visible corner hint and the hidden
    /// hold-to-reveal overlay. The reveal-glyph hint and the legend are both pre-built per
    /// device; [`update_controls_ui`] shows only the active device's.
    pub fn spawn_controls_ui<S: ControlScheme>(
        mut commands: Commands,
        asset_server: Res<AssetServer>,
    ) {
        let default_device = Device::default();

        // Corner hint (bottom-left), always visible: "[reveal glyph] Hold - Controls".
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(14.0),
                    left: Val::Px(14.0),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                },
                ControlsHintRoot,
            ))
            .with_children(|hint| {
                // One reveal-glyph node per device, only the active one shown — same pattern
                // as the legend columns, so an icon OR a text-keycap reveal glyph both work.
                for device in [Device::KeyboardMouse, Device::Gamepad] {
                    let Some(glyph) = reveal_glyph::<S>(device) else {
                        continue;
                    };
                    hint.spawn((
                        Node {
                            display: if device == default_device {
                                Display::Flex
                            } else {
                                Display::None
                            },
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        HintGlyphFor(device),
                    ))
                    .with_children(|slot| spawn_glyph(slot, &asset_server, glyph));
                }
                hint.spawn((
                    Text::new("Hold - Controls"),
                    TextFont {
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.85, 0.85, 0.85)),
                ));
            });

        // Overlay panel (centered, dark backdrop), hidden until the reveal control is held.
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Percent(18.0),
                    left: Val::Percent(34.0),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(24.0)),
                    row_gap: Val::Px(8.0),
                    display: Display::None,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.82)),
                ControlsOverlayRoot,
            ))
            .with_children(|overlay| {
                overlay.spawn((
                    Text::new("Controls"),
                    TextFont {
                        font_size: 26.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 1.0, 1.0)),
                ));
                spawn_legend_column::<S>(
                    overlay,
                    &asset_server,
                    Device::KeyboardMouse,
                    default_device == Device::KeyboardMouse,
                );
                spawn_legend_column::<S>(
                    overlay,
                    &asset_server,
                    Device::Gamepad,
                    default_device == Device::Gamepad,
                );
            });
    }

    /// Detect the active input device from this frame's input. A pressed key or moved mouse
    /// → keyboard/mouse; any pad button or past-deadzone stick → gamepad. Last input wins.
    /// Reads edges (`just_pressed`/motion), not held state, so holding a key doesn't pin the
    /// device against a pad the player picks up. Not generic — a pure input scan.
    pub fn track_active_device(
        keys: Res<ButtonInput<KeyCode>>,
        mouse_buttons: Res<ButtonInput<MouseButton>>,
        mouse_motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
        gamepads: Query<&Gamepad>,
        mut device: ResMut<ActiveDevice>,
    ) {
        let kb_active = keys.get_just_pressed().next().is_some()
            || mouse_buttons.get_just_pressed().next().is_some()
            || mouse_motion.delta != Vec2::ZERO;
        let pad_active = gamepads.iter().any(|gp| {
            gp.get_just_pressed().next().is_some()
                || gp.left_stick().length() > PAD_STICK_DEADZONE
                || gp.right_stick().length() > PAD_STICK_DEADZONE
        });
        // Pad checked last so simultaneous input favors the pad the player is holding.
        if kb_active {
            device.0 = Device::KeyboardMouse;
        }
        if pad_active {
            device.0 = Device::Gamepad;
        }
    }

    /// Whether the reveal control is currently held — any of its keyboard keys down OR any
    /// of its pad buttons down on any connected pad. Reads the scheme's map via the
    /// [`ControlInput`] glue, so the hint advertises exactly the control that opens the panel.
    fn reveal_held<S: ControlInput>(
        keys: &ButtonInput<KeyCode>,
        gamepads: &Query<&Gamepad>,
    ) -> bool {
        let Some(e) = entry::<S>(S::reveal_action()) else {
            return false;
        };
        let key_down = e
            .keyboard
            .keys
            .iter()
            .filter_map(|&k| S::key_code(k))
            .any(|k| keys.pressed(k));
        let pad_down = e
            .pad
            .buttons
            .iter()
            .filter_map(|&p| S::gamepad_button(p))
            .any(|b| gamepads.iter().any(|gp| gp.pressed(b)));
        key_down || pad_down
    }

    /// Each frame: show the overlay iff the reveal control is held, and show only the active
    /// device's legend column and hint glyph. Pure client UI — no sim contact. The three
    /// `&mut Node` queries carry mutual `Without` filters so Bevy proves them disjoint.
    // The query filter tuples are the disjointness proof, not gratuitous complexity — Bevy
    // needs them spelled out, so the lint doesn't apply here.
    #[allow(clippy::type_complexity)]
    pub fn update_controls_ui<S: ControlInput>(
        keys: Res<ButtonInput<KeyCode>>,
        gamepads: Query<&Gamepad>,
        device: Res<ActiveDevice>,
        force_reveal: Res<ForceRevealControls>,
        mut overlay: Query<
            &mut Node,
            (
                With<ControlsOverlayRoot>,
                Without<LegendColumn>,
                Without<HintGlyphFor>,
            ),
        >,
        mut columns: Query<
            (&LegendColumn, &mut Node),
            (Without<ControlsOverlayRoot>, Without<HintGlyphFor>),
        >,
        mut hints: Query<
            (&HintGlyphFor, &mut Node),
            (Without<ControlsOverlayRoot>, Without<LegendColumn>),
        >,
    ) {
        let revealed = force_reveal.0 || reveal_held::<S>(&keys, &gamepads);

        if let Ok(mut node) = overlay.single_mut() {
            node.display = if revealed {
                Display::Flex
            } else {
                Display::None
            };
        }

        let show_for = |d: Device| {
            if d == device.0 {
                Display::Flex
            } else {
                Display::None
            }
        };
        for (col, mut node) in &mut columns {
            node.display = show_for(col.0);
        }
        for (hint, mut node) in &mut hints {
            node.display = show_for(hint.0);
        }
    }

    /// Convenience plugin for the always-on case (the demo): spawns the controls UI at
    /// Startup and runs the device tracker + overlay update every frame. An app with a
    /// gated lifecycle (GCR spawns at its Playing transition) wires the systems directly
    /// instead.
    pub struct ControlsOverlayPlugin<S>(std::marker::PhantomData<fn() -> S>);

    impl<S> Default for ControlsOverlayPlugin<S> {
        fn default() -> Self {
            Self(std::marker::PhantomData)
        }
    }

    impl<S: ControlInput> Plugin for ControlsOverlayPlugin<S> {
        fn build(&self, app: &mut App) {
            app.init_resource::<ActiveDevice>()
                .insert_resource(ForceRevealControls(false))
                .add_systems(Startup, spawn_controls_ui::<S>)
                .add_systems(
                    Update,
                    (track_active_device, update_controls_ui::<S>).chain(),
                );
        }
    }
}

#[cfg(feature = "render")]
pub use overlay::{
    ActiveDevice, ControlInput, ControlsOverlayPlugin, ForceRevealControls, spawn_controls_ui,
    track_active_device, update_controls_ui,
};

#[cfg(feature = "render")]
pub(crate) use overlay::PAD_STICK_DEADZONE;

/// The headless/debug overlay override, from the rl env convention shared by every render
/// bin: `RL_SHOW_CONTROLS=1` forces the (normally hold-to-reveal) overlay open and
/// `RL_SHOW_CONTROLS_PAD=1` selects the gamepad column — so one windowless screenshot can
/// record either device's legend with no live input. One source so the contract can't drift
/// between bins (the demo's and GCR's screenshot paths both call it). Inert when unset.
#[cfg(feature = "render")]
pub(crate) fn reveal_overrides_from_env() -> (ForceRevealControls, ActiveDevice) {
    (
        ForceRevealControls(std::env::var_os("RL_SHOW_CONTROLS").is_some()),
        ActiveDevice(if std::env::var_os("RL_SHOW_CONTROLS_PAD").is_some() {
            Device::Gamepad
        } else {
            Device::KeyboardMouse
        }),
    )
}
