use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    KeyboardMouse,
    Gamepad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    Icon(&'static str),
    Label(&'static str),
}

pub trait ControlScheme: 'static + Send + Sync {
    type Action: Copy + PartialEq + Debug;
    type Key: Copy + PartialEq;
    type Pad: Copy + PartialEq;
    type Mouse: Copy + PartialEq;
    type Context: Copy + PartialEq + Eq + Debug + Default + Send + Sync + 'static;

    fn bindings() -> &'static [Binding<Self>];

    fn contexts() -> &'static [Self::Context];

    fn context_rows(ctx: Self::Context) -> &'static [ContextRow<Self>];

    fn context_label(ctx: Self::Context) -> &'static str;

    fn context_id(ctx: Self::Context) -> &'static str;

    /// Resolve a canonical [`context_id`](ControlScheme::context_id) back to its context.
    /// `None` for an unknown id — which `--show-controls-context` reports as an error
    /// naming [`contexts`](ControlScheme::contexts), never a silent default (rl#275).
    fn context_from_id(id: &str) -> Option<Self::Context>;

    fn reveal_action() -> Self::Action;

    fn key_glyph(key: Self::Key) -> Glyph;
    fn pad_glyph(pad: Self::Pad) -> Glyph;
    fn mouse_glyph(mouse: Self::Mouse) -> Glyph;
}

pub struct KbBinding<S: ControlScheme + ?Sized> {
    pub keys: &'static [S::Key],
    pub mouse: &'static [S::Mouse],
    pub hold: bool,
}

pub struct PadBinding<S: ControlScheme + ?Sized> {
    pub buttons: &'static [S::Pad],
    pub hold: bool,
}

impl<S: ControlScheme + ?Sized> KbBinding<S> {
    pub const NONE: Self = Self {
        keys: &[],
        mouse: &[],
        hold: false,
    };
    pub const fn new(keys: &'static [S::Key], mouse: &'static [S::Mouse]) -> Self {
        Self {
            keys,
            mouse,
            hold: false,
        }
    }
    pub const fn hold(keys: &'static [S::Key], mouse: &'static [S::Mouse]) -> Self {
        Self {
            keys,
            mouse,
            hold: true,
        }
    }
}

impl<S: ControlScheme + ?Sized> PadBinding<S> {
    pub const fn new(buttons: &'static [S::Pad]) -> Self {
        Self {
            buttons,
            hold: false,
        }
    }
    pub const fn hold(buttons: &'static [S::Pad]) -> Self {
        Self {
            buttons,
            hold: true,
        }
    }
}

pub struct Binding<S: ControlScheme + ?Sized> {
    pub action: S::Action,
    pub keyboard: KbBinding<S>,
    pub pad: PadBinding<S>,
}

impl<S: ControlScheme + ?Sized> Binding<S> {
    pub fn is_hold(&self, device: Device) -> bool {
        match device {
            Device::KeyboardMouse => self.keyboard.hold,
            Device::Gamepad => self.pad.hold,
        }
    }

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

pub struct ContextRow<S: ControlScheme + ?Sized> {
    pub action: S::Action,
    pub label: &'static str,
}

pub fn binding<S: ControlScheme + ?Sized>(action: S::Action) -> Option<&'static Binding<S>> {
    S::bindings().iter().find(|b| b.action == action)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegendLine {
    pub label: &'static str,
    pub hold: bool,
    pub glyphs: Vec<Glyph>,
}

pub fn legend<S: ControlScheme + ?Sized>(ctx: S::Context, device: Device) -> Vec<LegendLine> {
    S::context_rows(ctx)
        .iter()
        .filter_map(|row| {
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

pub fn reveal_glyph<S: ControlScheme + ?Sized>(device: Device) -> Option<Glyph> {
    binding::<S>(S::reveal_action()).and_then(|b| b.glyphs(device).first().copied())
}

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
    assert!(
        all_contexts.contains(&S::Context::default()),
        "Context::default() {:?} is not in contexts()",
        S::Context::default()
    );
    for &ctx in all_contexts {
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
    for device in [Device::KeyboardMouse, Device::Gamepad] {
        assert!(
            reveal_glyph::<S>(device).is_some(),
            "reveal action has no glyph on {device:?}"
        );
    }
}

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

#[cfg(feature = "render")]
mod overlay {
    use super::*;
    use bevy::prelude::*;

    /// The Bevy-input glue a scheme needs so its DISCRETE controls can be polled from the
    /// binding table — the overlay's reveal control and any tap verb an app dispatches via
    /// [`key_codes_for`]/[`gamepad_buttons_for`]. Analog/multi-role inputs (sticks, mouse
    /// motion, wheel, directional D-pad pairs) map to NO key — they're read via their own
    /// axis/multi-input APIs in the app's systems.
    ///
    /// A key token maps to a SLICE, not one `KeyCode`: one legend chip may legitimately
    /// stand for several physical keys (Enter and the numpad's Enter are one control to
    /// the player), and the alternative — a second token per synonym — would print a
    /// duplicate chip in the legend.
    pub trait ControlInput: ControlScheme {
        fn key_codes(key: Self::Key) -> &'static [KeyCode];
        fn gamepad_button(pad: Self::Pad) -> Option<GamepadButton>;
    }

    /// Every `KeyCode` bound to `action`, resolved from the scheme's binding table through
    /// its [`ControlInput`] glue — so a handler polls exactly the keys the legend shows;
    /// rebind the table and the poll and legend move together (the no-drift round-trip).
    /// Tokens that map to no key (analog/multi-key) drop out. An iterator, not a `Vec`:
    /// callers just `.any(...)` each frame, so no heap alloc.
    pub fn key_codes_for<S: ControlInput>(action: S::Action) -> impl Iterator<Item = KeyCode> {
        binding::<S>(action)
            .into_iter()
            .flat_map(|b| b.keyboard.keys.iter().copied())
            .flat_map(|k| S::key_codes(k).iter().copied())
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

    /// Whether `action` was just pressed this frame — on any of its bound keys or any bound
    /// button of any connected pad. THE tap-verb dispatch: apps route their discrete verbs
    /// through this one composition (the demo's verbs, GCR's render-mode cycle), so the
    /// composition itself can't drift per app. Reads with genuinely per-device semantics
    /// (GCR's kb-tap vs pad-hold quit, pad-only held actions) legitimately stay app-side.
    pub fn just_pressed<S: ControlInput>(
        action: S::Action,
        keys: &ButtonInput<KeyCode>,
        gamepads: &Query<&Gamepad>,
    ) -> bool {
        key_codes_for::<S>(action).any(|k| keys.just_pressed(k))
            || gamepads
                .iter()
                .any(|gp| gamepad_buttons_for::<S>(action).any(|b| gp.just_pressed(b)))
    }

    /// The held twin of [`just_pressed`] — any bound key or pad button currently DOWN.
    /// (The overlay's hold-to-reveal read.)
    pub fn pressed<S: ControlInput>(
        action: S::Action,
        keys: &ButtonInput<KeyCode>,
        gamepads: &Query<&Gamepad>,
    ) -> bool {
        key_codes_for::<S>(action).any(|k| keys.pressed(k))
            || gamepads
                .iter()
                .any(|gp| gamepad_buttons_for::<S>(action).any(|b| gp.pressed(b)))
    }

    #[derive(Resource, Clone, Copy, Default)]
    pub struct ActiveDevice(pub Device);

    /// The context whose legend the overlay is showing.
    ///
    /// PINNABLE: `--show-controls-context` fixes it for the life of the surface, and the pin
    /// is enforced INSIDE [`ActiveContext::sync`] — the only way to drive the context from live
    /// state. A surface therefore cannot forget to honor it; honoring used to be a convention
    /// each caller had to remember, gated on its own separate read of the knob (rl#275).
    #[derive(Resource)]
    pub struct ActiveContext<S: ControlScheme> {
        ctx: S::Context,
        pinned: bool,
    }

    impl<S: ControlScheme> Default for ActiveContext<S> {
        fn default() -> Self {
            Self {
                ctx: S::Context::default(),
                pinned: false,
            }
        }
    }

    impl<S: ControlScheme> ActiveContext<S> {
        pub(crate) fn pinned_to(ctx: S::Context, pinned: bool) -> Self {
            Self { ctx, pinned }
        }

        pub fn get(&self) -> S::Context {
            self.ctx
        }

        /// Retarget the legend from live state (the vehicle you're in, the menu you're on).
        /// A no-op while pinned: an evidence shot keeps the legend it asked for.
        pub fn sync(&mut self, ctx: S::Context) {
            if !self.pinned && self.ctx != ctx {
                self.ctx = ctx;
            }
        }
    }

    #[derive(Resource, Clone, Copy, Default)]
    pub struct ForceRevealControls(pub bool);

    /// Whether the overlay is showing THIS frame — written by [`update_controls_ui`], the
    /// one place that decides it. Readers (e.g. a menu yielding the screen to the overlay)
    /// take this instead of re-deriving reveal from the inputs, so they can't drift from
    /// the real visibility.
    #[derive(Resource, Clone, Copy, Default)]
    pub struct ControlsRevealed(pub bool);

    #[derive(Component)]
    pub struct ContextHintLabel;

    #[derive(Component)]
    pub struct ContextHeading;

    #[derive(Component)]
    pub struct ControlsOverlayRoot;

    #[derive(Component)]
    pub struct LegendColumn<S: ControlScheme> {
        ctx: S::Context,
        device: Device,
    }

    #[derive(Component)]
    pub struct HintGlyphFor(Device);

    const GLYPH_PX: f32 = 30.0;

    pub const PAD_STICK_DEADZONE: f32 = 0.15;

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

    pub fn spawn_controls_ui<S: ControlScheme>(
        mut commands: Commands,
        asset_server: Res<AssetServer>,
    ) {
        crate::assets::warn_missing_glyphs(super::icon_asset_paths::<S>());

        let default_device = Device::default();
        let default_ctx = S::Context::default();
        let default_label = S::context_label(default_ctx);

        commands
            .spawn(Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(14.0),
                left: Val::Px(14.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            })
            .with_children(|hint| {
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
        if kb_active {
            device.0 = Device::KeyboardMouse;
        }
        if pad_active {
            device.0 = Device::Gamepad;
        }
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn update_controls_ui<S: ControlInput>(
        keys: Res<ButtonInput<KeyCode>>,
        gamepads: Query<&Gamepad>,
        device: Res<ActiveDevice>,
        context: Res<ActiveContext<S>>,
        force_reveal: Res<ForceRevealControls>,
        mut revealed_out: ResMut<ControlsRevealed>,
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
        let revealed = force_reveal.0 || pressed::<S>(S::reveal_action(), &keys, &gamepads);
        revealed_out.0 = revealed;

        if let Ok(mut node) = overlay.single_mut() {
            node.display = if revealed {
                Display::Flex
            } else {
                Display::None
            };
        }

        for (col, mut node) in &mut columns {
            node.display = if col.ctx == context.get() && col.device == device.0 {
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

        let label = S::context_label(context.get());
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
                .init_resource::<ControlsRevealed>()
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
    ActiveContext, ActiveDevice, ControlInput, ControlsOverlayPlugin, ControlsOverlayRoot,
    ControlsRevealed, ForceRevealControls, gamepad_buttons_for, just_pressed, key_codes_for,
    pressed, spawn_controls_ui, track_active_device, update_controls_ui,
};

#[cfg(feature = "render")]
pub use overlay::PAD_STICK_DEADZONE;

/// DIAGNOSTIC force-knobs for the controls overlay: they pin what the legend shows, so an
/// evidence shot can prove a binding renders. Flattened by every surface that shows the legend.
#[cfg(feature = "render")]
#[derive(clap::Args, Debug, Clone, Default)]
pub struct ControlsOverlayArgs {
    /// Hold the controls legend open (it is otherwise revealed by holding the reveal control).
    #[arg(long, env = "RL_SHOW_CONTROLS", value_parser = clap::builder::FalseyValueParser::new())]
    pub show_controls: bool,

    /// Draw the legend with gamepad glyphs instead of keyboard/mouse.
    #[arg(long, env = "RL_SHOW_CONTROLS_PAD", value_parser = clap::builder::FalseyValueParser::new())]
    pub show_controls_pad: bool,

    /// Pin the legend to this control context, by its scheme id (e.g. `foot`, `plane`).
    /// Unset lets the surface's live state drive it.
    #[arg(long, env = "RL_SHOW_CONTROLS_CONTEXT", value_name = "ID")]
    pub show_controls_context: Option<String>,
}

/// [`ControlsOverlayArgs`] resolved against a scheme. The context id is checked at the
/// ENTRYPOINT, so [`install_overlay`] cannot fail and an unknown id can never fall through to
/// the default context — which would capture the wrong legend in a shot that reads as if the
/// override had taken (rl#275).
#[cfg(feature = "render")]
pub struct ControlsOverrides<S: ControlScheme> {
    reveal: bool,
    device: Device,
    /// `Some` IS the pin — one field, so the context and "is it pinned" cannot disagree.
    context: Option<S::Context>,
}

/// No force-knobs: what a surface that exposes none installs. Hand-written because deriving it
/// would demand `S: Default` — the scheme is a marker type, only its `Context` needs a default.
#[cfg(feature = "render")]
impl<S: ControlScheme> Default for ControlsOverrides<S> {
    fn default() -> Self {
        Self {
            reveal: false,
            device: Device::default(),
            context: None,
        }
    }
}

#[cfg(feature = "render")]
impl ControlsOverlayArgs {
    pub fn resolve<S: ControlScheme>(&self) -> Result<ControlsOverrides<S>, String> {
        let context = self
            .show_controls_context
            .as_deref()
            .map(|id| {
                S::context_from_id(id).ok_or_else(|| {
                    let known: Vec<&str> =
                        S::contexts().iter().map(|&c| S::context_id(c)).collect();
                    format!(
                        "--show-controls-context {id:?} is not a context of this surface; \
                         want one of: {}",
                        known.join(", ")
                    )
                })
            })
            .transpose()?;
        Ok(ControlsOverrides {
            reveal: self.show_controls,
            device: if self.show_controls_pad {
                Device::Gamepad
            } else {
                Device::KeyboardMouse
            },
            context,
        })
    }
}

/// Install the overlay with `overrides` applied — the ONE wiring of plugin + force-knobs,
/// shared by every surface that shows the legend (pass `&Default::default()` for a surface
/// that exposes no knobs). The resources land AFTER the plugin, replacing its defaults.
#[cfg(feature = "render")]
pub fn install_overlay<S: ControlInput>(
    app: &mut bevy::app::App,
    overrides: &ControlsOverrides<S>,
) {
    app.add_plugins(ControlsOverlayPlugin::<S>::default())
        .insert_resource(ForceRevealControls(overrides.reveal))
        .insert_resource(ActiveDevice(overrides.device))
        .insert_resource(ActiveContext::<S>::pinned_to(
            overrides.context.unwrap_or_default(),
            overrides.context.is_some(),
        ));
}
