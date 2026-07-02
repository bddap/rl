//! Reusable controls + hold-to-reveal-overlay framework, generic over an app's action set
//! AND its input contexts. One executable = one [`ControlScheme`] (its action enum, input
//! vocabulary, glyph art, binding table, and the per-context row lists); the framework turns
//! that into the on-screen control legend and the polished hold-to-reveal overlay. GCR
//! (`net::controls`) and the demo ([`crate::play`]) are disjoint apps with disjoint
//! verbs — each brings its own scheme; the overlay code below is shared.
//!
//! The no-drift guarantee, in two halves:
//! - **One binding table.** A scheme's [`ControlScheme::bindings`] is the ONE table that says
//!   which key/button triggers each action. The legend glyphs derive from it (via the
//!   `*_glyph` resolvers), and live input reads it too — both apps dispatch their discrete
//!   verbs via `key_codes_for`/`gamepad_buttons_for` over the [`ControlInput`] glue — so the
//!   round-trip is closed: rebind in the table, both the poll and the legend move together.
//!   The displayed bindings can't drift from the dispatched ones: the HUD resolves every key
//!   through the SAME `binding(action)` the input layer dispatches from.
//! - **Context as data.** The active control set is a CONTEXT (on-foot / piloting / …). Each
//!   context is a [`ContextRow`] list — the actions shown in that context, in legend order,
//!   each with the human label that's correct THERE ("Forward" on foot, "Throttle up" in the
//!   plane). The binding (the key) is shared; only the label and membership vary. The HUD
//!   renders [`legend`]`(ctx, device)` for the live [`ActiveContext`], so entering a vehicle
//!   re-derives the panel from the one table — a stale/parallel HUD is unrepresentable.
//!
//! Two layers, split by the `render` feature — exactly like the rest of the crate:
//! - The **pure core** ([`Device`], [`Glyph`], [`LegendLine`], [`Binding`], [`ContextRow`],
//!   the [`ControlScheme`] trait, [`legend`]/[`reveal_glyph`], [`assert_scheme_well_formed`])
//!   has NO Bevy dependency, so it compiles and unit-tests in the no-feature build.
//! - The **Bevy glue** (the `#[cfg(feature = "render")]` block) is the only place the typed
//!   inputs meet Bevy's input API and the only place the overlay UI lives.

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

/// An app's control scheme: its action enum, input vocabulary, glyph art, the binding table,
/// and the input contexts (each a [`ContextRow`] list). Implemented once per executable. The
/// associated input types ([`Key`](ControlScheme::Key)/[`Pad`](ControlScheme::Pad)/[`Mouse`](ControlScheme::Mouse))
/// make the scheme self-describing, so the legend's glyphs derive from the SAME typed
/// bindings the input reads — no drift.
pub trait ControlScheme: 'static + Send + Sync {
    /// The controllable verbs. The row key of [`ControlScheme::bindings`]; the per-app
    /// invariant test ([`assert_scheme_well_formed`]) proves every variant has exactly one
    /// binding.
    type Action: Copy + PartialEq + Debug;
    /// The app's keyboard vocabulary (a closed enum: only the keys it binds), so the binding
    /// and glyph tables are exhaustive and a typo can't name a nonexistent key.
    type Key: Copy + PartialEq;
    /// The app's gamepad vocabulary.
    type Pad: Copy + PartialEq;
    /// The app's mouse vocabulary (motion / buttons / wheel).
    type Mouse: Copy + PartialEq;
    /// The app's input CONTEXTS — the distinct control sets the player moves between
    /// (on-foot, piloting a plane, …). A single context for an app that never switches.
    /// `Default` is the context the overlay starts in (the foot/primary context).
    type Context: Copy + PartialEq + Eq + Debug + Default + Send + Sync + 'static;

    /// THE binding table: one row per action, the key/button that triggers it on each device.
    /// Context-INDEPENDENT — an action's binding is the same wherever it appears, so the key
    /// the legend shows is the key the input polls. Order is the canonical action order.
    fn bindings() -> &'static [Binding<Self>];

    /// Every context, in display order — drives pre-building the per-context legend columns
    /// and the well-formedness test.
    fn contexts() -> &'static [Self::Context];

    /// The actions visible in a context, in legend order, each with the label that's correct
    /// IN that context. The HUD for `ctx` is built by joining these rows with
    /// [`bindings`](ControlScheme::bindings) for glyphs — so the panel IS the live bindings,
    /// labeled for where you are.
    fn context_rows(ctx: Self::Context) -> &'static [ContextRow<Self>];

    /// Human name of a context — the overlay heading and the always-visible corner hint
    /// ("On foot", "Piloting plane"), so the player always knows which control set is live.
    fn context_label(ctx: Self::Context) -> &'static str;

    /// The canonical short id of a context (a slug for the `RL_SHOW_CONTROLS_CONTEXT`
    /// screenshot override, e.g. `"foot"`/`"plane"`). [`assert_scheme_well_formed`] proves
    /// every context round-trips `context_from_id(context_id(c)) == Some(c)`, so each context
    /// is reachable by a stable id — a new context can't be added screenshot-unreachable.
    fn context_id(ctx: Self::Context) -> &'static str;

    /// Resolve a canonical [`context_id`](ControlScheme::context_id) (from
    /// `RL_SHOW_CONTROLS_CONTEXT`) back to its context. `None` for an unknown id (the
    /// override then leaves the default context).
    fn context_from_id(id: &str) -> Option<Self::Context>;

    /// The action whose HOLD reveals the overlay (and whose glyph the corner hint shows).
    /// Bound once (context-independent) and present in every context's rows.
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

/// One row of a scheme's binding table: an action and the key/button that triggers it on
/// each device. NO label — the label lives in [`ContextRow`], because the same binding reads
/// differently in different contexts (the W key is "Forward" on foot, "Throttle up" in the
/// plane). This split is what makes the binding single-source while letting labels be
/// context-correct.
pub struct Binding<S: ControlScheme + ?Sized> {
    pub action: S::Action,
    pub keyboard: KbBinding<S>,
    pub pad: PadBinding<S>,
}

impl<S: ControlScheme + ?Sized> Binding<S> {
    /// Whether this binding is a hold on the given device.
    pub fn is_hold(&self, device: Device) -> bool {
        match device {
            Device::KeyboardMouse => self.keyboard.hold,
            Device::Gamepad => self.pad.hold,
        }
    }

    /// The ordered glyphs to display for this binding on `device` (empty if it has no binding
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

/// One row of a context's control list: an action shown in that context + the human label
/// for it THERE. The binding (glyphs/hold) is looked up from [`ControlScheme::bindings`] by
/// `action`, so the displayed KEY can't drift from the polled key (both resolve through the
/// one binding). The label's MEANING — that "Throttle up" really is what `move_forward` does
/// in the plane — is the one hand-maintained link the types don't enforce; a scheme pins the
/// non-obvious ones with a test (see GCR's plane-pitch-sign test).
pub struct ContextRow<S: ControlScheme + ?Sized> {
    pub action: S::Action,
    /// Short human label for the legend in this context (e.g. "Forward" / "Throttle up").
    pub label: &'static str,
}

/// Look up a scheme's binding for `action`. Total over the scheme's actions (every action has
/// exactly one binding — proven by [`assert_scheme_well_formed`]), so the `Option` is just
/// the lookup's honest shape.
pub fn binding<S: ControlScheme + ?Sized>(action: S::Action) -> Option<&'static Binding<S>> {
    S::bindings().iter().find(|b| b.action == action)
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

/// Build the legend for context `ctx` on `device`: one [`LegendLine`] per context row that
/// is BOUND on that device, in context-row order, labeled for `ctx`. Rows whose action has
/// no binding on the device are omitted (so the keyboard column doesn't list a pad-only
/// control). Joining the context rows with the binding table is what keeps the bindings the
/// only enumeration source — the legend can't drift from the live keys.
pub fn legend<S: ControlScheme + ?Sized>(ctx: S::Context, device: Device) -> Vec<LegendLine> {
    S::context_rows(ctx)
        .iter()
        .filter_map(|row| {
            // Every context row's action has a binding (proven by `assert_scheme_well_formed`),
            // so a miss is a malformed scheme — panic loudly rather than silently dropping the
            // line from the HUD (which would read as "that control doesn't exist").
            let b = binding::<S>(row.action).expect(
                "context row's action has no binding — assert_scheme_well_formed missed it",
            );
            let glyphs = b.glyphs(device);
            if glyphs.is_empty() {
                return None;
            }
            Some(LegendLine {
                label: row.label,
                hold: b.is_hold(device),
                glyphs,
            })
        })
        .collect()
}

/// The single glyph for the reveal control on `device`, for the corner hint. Context-
/// independent: the reveal control is bound once and shown in every context. The reveal
/// action binds at least one control per device (its first glyph), so this never returns
/// `None` for a well-formed scheme; the `Option` documents the invariant rather than
/// inventing a fallback.
pub fn reveal_glyph<S: ControlScheme + ?Sized>(device: Device) -> Option<Glyph> {
    binding::<S>(S::reveal_action()).and_then(|b| b.glyphs(device).first().copied())
}

/// Per-app well-formedness check, called from each scheme's own `#[test]` with its
/// exhaustive action + context lists. Proves the no-drift invariants the types alone can't:
/// every action has exactly one binding and is bound on some device; every context's rows
/// reference real actions, are bound on some device, and INCLUDE the reveal action (so the
/// overlay is openable everywhere); and the reveal control resolves to a hint glyph on both
/// devices. The compiler-side half — that a new `Action`/`Context` variant can't be added
/// without showing up here — is each app's exhaustive `match` producing these lists.
pub fn assert_scheme_well_formed<S: ControlScheme + ?Sized>(
    all_actions: &[S::Action],
    all_contexts: &[S::Context],
) {
    for &a in all_actions {
        let n = S::bindings().iter().filter(|b| b.action == a).count();
        assert_eq!(n, 1, "{a:?} has {n} bindings; want exactly 1");
        let b = binding::<S>(a).expect("just checked exactly one binding exists");
        assert!(
            !b.glyphs(Device::KeyboardMouse).is_empty() || !b.glyphs(Device::Gamepad).is_empty(),
            "{a:?} is bound on no device (would be invisible/unusable)"
        );
    }
    assert_eq!(
        S::bindings().len(),
        all_actions.len(),
        "the binding table has rows for actions not in the exhaustive list (a stale/dup row)"
    );
    assert!(
        !all_contexts.is_empty(),
        "a scheme needs at least one context"
    );
    // The default context (the overlay's spawn/idle state) must be a real, listed context —
    // else the HUD boots showing a context that isn't in `contexts()` and shows no column.
    assert!(
        all_contexts.contains(&S::Context::default()),
        "Context::default() {:?} is not in contexts()",
        S::Context::default()
    );
    for &ctx in all_contexts {
        // Every context is reachable by a stable id (the screenshot override), so a new
        // context can't be added that the evidence harness silently can't target.
        assert_eq!(
            S::context_from_id(S::context_id(ctx)),
            Some(ctx),
            "context {ctx:?} does not round-trip through context_id/context_from_id"
        );
        let rows = S::context_rows(ctx);
        assert!(
            !rows.is_empty(),
            "context {ctx:?} shows no controls (an empty legend)"
        );
        for row in rows {
            let b = binding::<S>(row.action)
                .unwrap_or_else(|| panic!("context {ctx:?} row {:?} has no binding", row.action));
            assert!(
                !b.glyphs(Device::KeyboardMouse).is_empty()
                    || !b.glyphs(Device::Gamepad).is_empty(),
                "context {ctx:?} shows {:?}, which is bound on no device",
                row.action
            );
        }
        assert!(
            rows.iter().any(|r| r.action == S::reveal_action()),
            "context {ctx:?} omits the reveal control — the overlay couldn't be opened there"
        );
    }
    // The reveal action must resolve to a hint glyph on both devices (the corner hint).
    for device in [Device::KeyboardMouse, Device::Gamepad] {
        assert!(
            reveal_glyph::<S>(device).is_some(),
            "reveal action has no glyph on {device:?}"
        );
    }
}

/// Every `controls/…` icon asset path scheme `S` can surface, across all its bindings and
/// both devices, deduped in first-seen order. Drives the startup glyph-presence check
/// ([`crate::assets::warn_missing_glyphs`]) so a new binding with a typo'd or unvendored
/// icon path is logged loudly, instead of just drawing a blank box on the overlay.
pub fn icon_asset_paths<S: ControlScheme + ?Sized>() -> Vec<&'static str> {
    let mut paths = Vec::new();
    for b in S::bindings() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for glyph in b.glyphs(device) {
                if let Glyph::Icon(p) = glyph
                    && !paths.contains(&p)
                {
                    paths.push(p);
                }
            }
        }
    }
    paths
}

// ---------------------------------------------------------------------------
// Bevy glue — the typed inputs meet Bevy's input API, and the overlay UI lives here.
// Render-only: Bevy's KeyCode/GamepadButton exist only under the `render` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "render")]
mod overlay {
    use super::*;
    use bevy::prelude::*;

    /// The Bevy-input glue a scheme needs so its DISCRETE controls can be polled from the
    /// binding table — the overlay's reveal control and any tap verb an app dispatches via
    /// [`key_codes_for`]/[`gamepad_buttons_for`]. Analog/multi-role inputs (sticks, mouse
    /// motion, wheel, directional D-pad pairs) return `None` — they're read via their own
    /// axis/multi-input APIs in the app's systems.
    pub trait ControlInput: ControlScheme {
        /// The Bevy `KeyCode` for a key, if it maps to one (analog/non-key inputs: `None`).
        fn key_code(key: Self::Key) -> Option<KeyCode>;
        /// The Bevy `GamepadButton` for a pad control, if it's a discrete button (sticks /
        /// D-pad read via their own axis/multi-button APIs: `None`).
        fn gamepad_button(pad: Self::Pad) -> Option<GamepadButton>;
    }

    /// Every `KeyCode` bound to `action`, resolved from the scheme's binding table through
    /// its [`ControlInput`] glue — so a handler polls exactly the keys the legend shows;
    /// rebind the table and the poll and legend move together (the no-drift round-trip).
    /// Tokens with no single `KeyCode` (analog/multi-key) drop out. An iterator, not a
    /// `Vec`: callers just `.any(...)` each frame, so no heap alloc.
    pub fn key_codes_for<S: ControlInput>(action: S::Action) -> impl Iterator<Item = KeyCode> {
        binding::<S>(action)
            .into_iter()
            .flat_map(|b| b.keyboard.keys.iter().copied())
            .filter_map(S::key_code)
    }

    /// Every pad `GamepadButton` bound to `action` — the pad half of [`key_codes_for`].
    pub fn gamepad_buttons_for<S: ControlInput>(
        action: S::Action,
    ) -> impl Iterator<Item = GamepadButton> {
        binding::<S>(action)
            .into_iter()
            .flat_map(|b| b.pad.buttons.iter().copied())
            .filter_map(S::gamepad_button)
    }

    /// The input device the player is currently using, refreshed each frame by
    /// [`track_active_device`]. Drives which legend column + hint glyph show. Pure client UI.
    #[derive(Resource, Clone, Copy, Default)]
    pub struct ActiveDevice(pub Device);

    /// The live input CONTEXT for scheme `S` — the control set whose legend the overlay
    /// shows and whose name the corner hint displays. The app drives it (GCR maps its
    /// `LocalVehicle` to a context each frame); the overlay only reads it. Pure client UI —
    /// it never touches the deterministic sim. Defaults to the scheme's default context.
    #[derive(Resource, Clone, Copy)]
    pub struct ActiveContext<S: ControlScheme>(pub S::Context);

    impl<S: ControlScheme> Default for ActiveContext<S> {
        fn default() -> Self {
            Self(S::Context::default())
        }
    }

    /// Force the reveal overlay open regardless of input — for a HEADLESS screenshot, which
    /// has no live keyboard/pad to hold the reveal control. Defaults false (the windowed
    /// client stays hold-to-reveal); a screenshot app sets it true for an evidence frame.
    #[derive(Resource, Clone, Copy, Default)]
    pub struct ForceRevealControls(pub bool);

    /// Marks the always-visible corner hint root ("[context] · Hold [glyph] Controls").
    #[derive(Component)]
    pub struct ControlsHintRoot;

    /// Marks the context-name text in the corner hint (always visible) — updated each frame
    /// to the active context's label so the player always knows which control set is live.
    #[derive(Component)]
    pub struct ContextHintLabel;

    /// Marks the context-name heading at the top of the reveal panel — updated each frame to
    /// the active context's label (e.g. "Piloting plane"), so the open overlay names the
    /// context too.
    #[derive(Component)]
    pub struct ContextHeading;

    /// Marks the hold-to-reveal overlay root (the dark panel). Toggled between
    /// [`Display::None`] and [`Display::Flex`] by [`update_controls_ui`].
    #[derive(Component)]
    pub struct ControlsOverlayRoot;

    /// One (context, device) legend container inside the overlay (pre-built once each). Only
    /// the active context's active-device container is shown — pre-building every combination
    /// avoids rebuilding child entities on a context switch or device pickup. Holds the
    /// context VALUE (not an index into `contexts()`), so the shown-column test is a direct
    /// `col.ctx == active` compare with nothing to fall out of sync.
    #[derive(Component)]
    pub struct LegendColumn<S: ControlScheme> {
        ctx: S::Context,
        device: Device,
    }

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
    pub const PAD_STICK_DEADZONE: f32 = 0.15;

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

    /// Spawn one (context, device) legend column (icon/label + label rows) as a child of the
    /// overlay. Each row comes straight from [`legend`]`(ctx, device)`, so the panel IS the
    /// live bindings, labeled for that context.
    fn spawn_legend_column<S: ControlScheme>(
        parent: &mut ChildSpawnerCommands,
        asset_server: &AssetServer,
        ctx: S::Context,
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
                LegendColumn::<S> { ctx, device },
            ))
            .with_children(|col| {
                for line in legend::<S>(ctx, device) {
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

    /// Spawn the controls UI for scheme `S`: the always-visible corner hint (the active
    /// context name + "Hold [glyph] Controls") and the hidden hold-to-reveal overlay. The
    /// reveal-glyph hint and one legend column per (context, device) are pre-built;
    /// [`update_controls_ui`] shows only the active context's active-device column and keeps
    /// the context name in sync.
    pub fn spawn_controls_ui<S: ControlScheme>(
        mut commands: Commands,
        asset_server: Res<AssetServer>,
    ) {
        // WARN loudly at spawn for any glyph this overlay will request that isn't on disk
        // under the resolved asset root — every overlay path funnels through here, so this
        // one check covers the game and the demo (and any future scheme). A missing icon is
        // decorative HUD: it degrades to a blank slot (bevy renders a missing handle as
        // nothing) rather than aborting the match — but it's logged with its path so the
        // gap isn't silent.
        crate::assets::warn_missing_glyphs(super::icon_asset_paths::<S>());

        let default_device = Device::default();
        let default_ctx = S::Context::default();
        let default_label = S::context_label(default_ctx);

        // Corner hint (bottom-left), always visible: "[context] · [reveal glyph] Hold - Controls".
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
                // The active context name, kept current by `update_controls_ui` — so which
                // control set is live is visible WITHOUT opening the overlay.
                hint.spawn((
                    Text::new(default_label),
                    TextFont {
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.95, 0.6)),
                    ContextHintLabel,
                ));
                hint.spawn((
                    Text::new("·"),
                    TextFont {
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.6, 0.6, 0.6)),
                ));
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

        // Hold-to-reveal overlay, hidden until the reveal control is held. The root is a
        // full-screen flex layer that CENTERS its one child (the dark content panel) — so the
        // panel is centered for any content size and any UiScale, and can never flow off a
        // small screen like the Steam Deck's 1280×800 (the old fixed top/left-percent anchor
        // pinned the panel's CORNER, so a wide legend or a scaled-up UI ran off-screen). The
        // `max_*` caps keep the panel inside the viewport as a backstop. Display toggles on
        // this root, unchanged — `update_controls_ui` still flips one node.
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(0.0),
                    left: Val::Px(0.0),
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    display: Display::None,
                    ..default()
                },
                ControlsOverlayRoot,
            ))
            .with_children(|root| {
                root.spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        max_width: Val::Percent(92.0),
                        max_height: Val::Percent(92.0),
                        padding: UiRect::all(Val::Px(24.0)),
                        row_gap: Val::Px(8.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.82)),
                ))
                .with_children(|overlay| {
                    // Context name, big — the open panel names the live control set.
                    overlay.spawn((
                        Text::new(default_label),
                        TextFont {
                            font_size: 26.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.95, 0.6)),
                        ContextHeading,
                    ));
                    overlay.spawn((
                        Text::new("Controls"),
                        TextFont {
                            font_size: 16.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.7, 0.7, 0.7)),
                    ));
                    // One legend column per (context, device); only the default pair starts shown.
                    for &ctx in S::contexts() {
                        for device in [Device::KeyboardMouse, Device::Gamepad] {
                            spawn_legend_column::<S>(
                                overlay,
                                &asset_server,
                                ctx,
                                device,
                                ctx == default_ctx && device == default_device,
                            );
                        }
                    }
                });
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
    /// of its pad buttons down on any connected pad. Reads the scheme's binding table via the
    /// [`ControlInput`] glue, so the hint advertises exactly the control that opens the panel.
    fn reveal_held<S: ControlInput>(
        keys: &ButtonInput<KeyCode>,
        gamepads: &Query<&Gamepad>,
    ) -> bool {
        key_codes_for::<S>(S::reveal_action()).any(|k| keys.pressed(k))
            || gamepad_buttons_for::<S>(S::reveal_action())
                .any(|btn| gamepads.iter().any(|gp| gp.pressed(btn)))
    }

    /// Each frame: show the overlay iff the reveal control is held; show only the active
    /// context's active-device legend column and the active device's hint glyph; and keep
    /// the context name (corner hint + panel heading) in sync with [`ActiveContext`]. Pure
    /// client UI — no sim contact. The `&mut Node`/`&mut Text` queries carry mutual `Without`
    /// filters so Bevy proves them disjoint.
    // The query filter tuples are the disjointness proof, not gratuitous complexity — Bevy
    // needs them spelled out, so the lint doesn't apply here. Likewise the arg count: each is
    // a distinct system param (inputs, the two name sinks, the column/hint/overlay queries).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn update_controls_ui<S: ControlInput>(
        keys: Res<ButtonInput<KeyCode>>,
        gamepads: Query<&Gamepad>,
        device: Res<ActiveDevice>,
        context: Res<ActiveContext<S>>,
        force_reveal: Res<ForceRevealControls>,
        mut overlay: Query<
            &mut Node,
            (
                With<ControlsOverlayRoot>,
                Without<LegendColumn<S>>,
                Without<HintGlyphFor>,
            ),
        >,
        mut columns: Query<
            (&LegendColumn<S>, &mut Node),
            (Without<ControlsOverlayRoot>, Without<HintGlyphFor>),
        >,
        mut hints: Query<
            (&HintGlyphFor, &mut Node),
            (Without<ControlsOverlayRoot>, Without<LegendColumn<S>>),
        >,
        mut headings: Query<&mut Text, (With<ContextHeading>, Without<ContextHintLabel>)>,
        mut hint_labels: Query<&mut Text, (With<ContextHintLabel>, Without<ContextHeading>)>,
    ) {
        let revealed = force_reveal.0 || reveal_held::<S>(&keys, &gamepads);

        if let Ok(mut node) = overlay.single_mut() {
            node.display = if revealed {
                Display::Flex
            } else {
                Display::None
            };
        }

        // Show the column for the live (context, device); compare the context VALUE directly.
        for (col, mut node) in &mut columns {
            node.display = if col.ctx == context.0 && col.device == device.0 {
                Display::Flex
            } else {
                Display::None
            };
        }
        for (hint, mut node) in &mut hints {
            node.display = if hint.0 == device.0 {
                Display::Flex
            } else {
                Display::None
            };
        }

        // Keep the context name current wherever it shows.
        let label = S::context_label(context.0);
        for mut text in &mut headings {
            if text.0 != label {
                text.0 = label.to_string();
            }
        }
        for mut text in &mut hint_labels {
            if text.0 != label {
                text.0 = label.to_string();
            }
        }
    }

    /// Convenience plugin for the always-on, single-context case (the demo): spawns the
    /// controls UI at Startup and runs the device tracker + overlay update every frame. An
    /// app that switches contexts (GCR) wires the systems directly and drives
    /// [`ActiveContext`] itself.
    pub struct ControlsOverlayPlugin<S>(std::marker::PhantomData<fn() -> S>);

    impl<S> Default for ControlsOverlayPlugin<S> {
        fn default() -> Self {
            Self(std::marker::PhantomData)
        }
    }

    impl<S: ControlInput> Plugin for ControlsOverlayPlugin<S> {
        fn build(&self, app: &mut App) {
            app.init_resource::<ActiveDevice>()
                .init_resource::<ActiveContext<S>>()
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
    ActiveContext, ActiveDevice, ControlInput, ControlsOverlayPlugin, ForceRevealControls,
    gamepad_buttons_for, key_codes_for, spawn_controls_ui, track_active_device,
    update_controls_ui,
};

#[cfg(feature = "render")]
pub use overlay::PAD_STICK_DEADZONE;

/// The headless/debug overlay override, from the rl env convention shared by every render
/// bin: `RL_SHOW_CONTROLS=1` forces the (normally hold-to-reveal) overlay open,
/// `RL_SHOW_CONTROLS_PAD=1` selects the gamepad column, and `RL_SHOW_CONTROLS_CONTEXT=<id>`
/// selects which context's legend to show (via [`ControlScheme::context_from_id`]) — so one
/// windowless screenshot can record any context/device's legend with no live input. One
/// source so the contract can't drift between bins (the demo's and GCR's screenshot paths
/// both call it). Inert when unset (default context/device, overlay closed).
#[cfg(feature = "render")]
pub fn reveal_overrides_from_env<S: ControlScheme>()
-> (ForceRevealControls, ActiveDevice, ActiveContext<S>) {
    let ctx = std::env::var("RL_SHOW_CONTROLS_CONTEXT")
        .ok()
        .and_then(|id| S::context_from_id(&id))
        .unwrap_or_default();
    (
        ForceRevealControls(std::env::var_os("RL_SHOW_CONTROLS").is_some()),
        ActiveDevice(if std::env::var_os("RL_SHOW_CONTROLS_PAD").is_some() {
            Device::Gamepad
        } else {
            Device::KeyboardMouse
        }),
        ActiveContext(ctx),
    )
}
