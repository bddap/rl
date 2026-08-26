use anyhow::Result;
pub use iroh::EndpointId;

use crate::client::ClientSim;
use crate::formation::{self, FormationDriver};
use crate::membership::Role;
use crate::net_loop::{self, MatchResult, NetDriver};
use crate::telemetry::TelemetrySender;
use crate::transport::Session;

#[derive(Debug, Clone)]
pub enum StartChoice {
    Host,
    Join(Option<EndpointId>),
}

const NET_EXPECT: usize = 2;

/// Where a [`Formation`] is in its life. The session bind is pollable — async on web,
/// resolved-on-first-poll native ([`crate::transport::bind_session`]) — so the lobby
/// has a pre-session phase on BOTH platforms: one path, no wasm fork (rl#412).
enum FormState {
    /// The session is still binding; everything the lobby needs once it lands.
    Binding {
        pending: crate::transport::PendingSession,
        role: Role,
        collector: Option<EndpointId>,
    },
    Live {
        session: Session,
        // Boxed for variant-size parity with Binding (clippy::large_enum_variant).
        driver: Box<FormationDriver>,
        telemetry: Option<TelemetrySender>,
    },
}

/// A lobby formation the render loop PUMPS: the session and the formation core live
/// right here, every accessor reads them directly, and the frame loop's `poll` drives
/// the bind, beats, roster, and agreement.
pub struct Formation {
    /// Consumed into the [`NetDriver`] when the match forms; `None` after resolution.
    state: Option<FormState>,
    dial_code: Option<EndpointId>,
    cancelled: bool,
    seed: u64,
    stamp: crate::SyncStamp,
    pub hosting: bool,
}

impl Formation {
    /// One frame's pump. `None` while the lobby is still binding or forming.
    pub fn poll(&mut self) -> Option<Result<MatchResult>> {
        self.state.as_ref()?;
        if self.cancelled {
            println!("lobby cancelled by the player");
            if let Some(FormState::Live {
                session, telemetry, ..
            }) = self.state.take()
            {
                net_loop::shutdown(&session, telemetry);
            }
            // A cancelled Binding state needs no teardown call: dropping the pending
            // drops its eventual Session, whose own Drop closes the endpoint.
            return Some(Ok(MatchResult::Cancelled));
        }
        if let Some(FormState::Binding { pending, .. }) = self.state.as_mut() {
            let bound = pending.poll()?;
            let Some(FormState::Binding {
                role, collector, ..
            }) = self.state.take()
            else {
                unreachable!("matched Binding above")
            };
            match bound {
                Ok(session) => {
                    // Same frame: fall through and pump the fresh lobby below.
                    self.state = Some(open_lobby(
                        session,
                        role,
                        self.dial_code,
                        collector,
                        self.stamp,
                    ));
                }
                Err(e) => return Some(Err(e.context("binding the lobby session"))),
            }
        }
        let Some(FormState::Live {
            session,
            driver,
            telemetry,
        }) = self.state.as_mut()
        else {
            unreachable!("Binding resolved above, and an empty state returned early")
        };
        let outcome = driver.pump(session, telemetry.as_ref())?;
        let Some(FormState::Live {
            session, telemetry, ..
        }) = self.state.take()
        else {
            unreachable!("matched Live above")
        };
        Some(match outcome {
            Ok(formation::Formation::Agreed(frozen)) => Ok(MatchResult::Joined(
                net_loop::joined_from_frozen(session, telemetry, frozen, self.seed, self.stamp),
            )),
            Ok(formation::Formation::Alone) => {
                net_loop::shutdown(&session, telemetry);
                Ok(MatchResult::Alone)
            }
            Err(e) => {
                net_loop::shutdown(&session, telemetry);
                Err(e)
            }
        })
    }

    fn live_session(&self) -> Option<&Session> {
        match self.state.as_ref()? {
            FormState::Binding { .. } => None,
            FormState::Live { session, .. } => Some(session),
        }
    }

    fn live_driver(&self) -> Option<&FormationDriver> {
        match self.state.as_ref()? {
            FormState::Binding { .. } => None,
            FormState::Live { driver, .. } => Some(driver),
        }
    }

    /// This peer's own endpoint id — `None` while the session is still binding (the
    /// lobby UI shows the code once it lands) and after resolution consumed it.
    pub fn my_id(&self) -> Option<EndpointId> {
        self.live_session().map(Session::endpoint_id)
    }

    pub fn display_code(&self) -> Option<EndpointId> {
        if self.hosting {
            self.my_id()
        } else {
            self.dial_code
        }
    }

    pub fn roster(&self) -> Vec<EndpointId> {
        self.live_driver()
            .map(|d| d.roster().to_vec())
            .unwrap_or_default()
    }

    pub fn lobby_len(&self) -> usize {
        self.live_driver().map(|d| d.roster().len()).unwrap_or(0)
    }

    pub fn request_start(&mut self) {
        // A joiner's press is inert in the CORE (`Membership::set_starting` no-ops
        // outside host mode, rl#94 liveness) — no second guard here. A press while
        // still BINDING is inert too: the lobby UI can't render Start before the
        // session lands (lobby_len is 0), so there is nothing to arm yet.
        if let Some(FormState::Live { driver, .. }) = self.state.as_mut() {
            driver.set_starting();
        }
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}

impl Drop for Formation {
    fn drop(&mut self) {
        // A formation abandoned mid-lobby (the menu drops it on cancel without another
        // poll) still tears down gracefully — telemetry drained, endpoint closed. An
        // abandoned Binding state tears itself down (see the cancel arm in poll).
        if let Some(FormState::Live {
            session, telemetry, ..
        }) = self.state.take()
        {
            net_loop::shutdown(&session, telemetry);
        }
    }
}

/// Begin a lobby: kick off the session bind and return the pollable formation —
/// non-blocking and platform-free (the web bind rides the JS event loop; native's
/// resolves on the first poll). Errors, including a failed bind, surface through
/// [`Formation::poll`].
pub fn begin(
    choice: &StartChoice,
    seed: u64,
    telemetry: Option<EndpointId>,
    stamp: crate::SyncStamp,
) -> Formation {
    let (role, join) = match choice {
        StartChoice::Host => (Role::Host, None),
        StartChoice::Join(host) => (Role::Joiner, *host),
    };
    Formation {
        state: Some(FormState::Binding {
            pending: crate::transport::bind_session(),
            role,
            collector: telemetry,
        }),
        dial_code: join,
        cancelled: false,
        seed,
        stamp,
        hosting: matches!(role, Role::Host),
    }
}

/// The bound-session half of [`begin`]: fire the join dial, wire telemetry, open the
/// formation core.
fn open_lobby(
    session: Session,
    role: Role,
    join: Option<EndpointId>,
    collector: Option<EndpointId>,
    stamp: crate::SyncStamp,
) -> FormState {
    if let Some(host) = join {
        if host == session.endpoint_id() {
            tracing::warn!("join code is our own endpoint id — ignoring the self-dial");
        } else {
            // Fire-and-forget: a failed dial surfaces as a lobby that never fills (the
            // joiner cancels out), matching the discovery-may-still-find-them semantics.
            let _ = session.dial(host);
        }
    }
    let telemetry = net_loop::connect_telemetry(&session, collector);
    let driver = Box::new(FormationDriver::lobby(&session, role, NET_EXPECT, stamp));
    FormState::Live {
        session,
        driver,
        telemetry,
    }
}

pub struct ReadyMatch {
    pub client: ClientSim,
    pub net: Option<NetDriver>,
}

pub fn ready_from(result: MatchResult, seed: u64) -> Option<ReadyMatch> {
    match result {
        MatchResult::Joined(joined) => {
            let (client, net) = *joined;
            Some(ReadyMatch {
                client,
                net: Some(net),
            })
        }
        MatchResult::Alone => Some(solo_round(seed)),
        MatchResult::Cancelled => None,
    }
}

pub fn solo_round(seed: u64) -> ReadyMatch {
    ReadyMatch {
        client: formation::solo_client_for(seed),
        net: None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChooserItem {
    Host,
    /// Straight into a solo round — no session, no lobby, ZERO network. The one
    /// entry a browser build can take today, and an instant offline start on native.
    Solo,
    Join,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobbyItem {
    Start,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisconnectedItem {
    Rejoin,
    Leave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuInput {
    Up,
    Down,
    Confirm,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    None,
    Host,
    Join,
    StartNetworked,
    StartSolo,
    Cancel,
    Rejoin,
    /// The user declined the "Connection lost" rejoin offer — the app drops `last_host`
    /// so re-entering the menu lands on the chooser, not the dead offer.
    DismissRejoin,
    /// Close the whole app from the boot menu (rl#263 — the only way out used to be
    /// starting a round and quitting from inside it).
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuNav {
    Chooser { focus: ChooserItem },
    HostLobby { focus: LobbyItem },
    JoinLobby,
    Disconnected { focus: DisconnectedItem },
    Rejoining,
}

impl Default for MenuNav {
    fn default() -> Self {
        Self::new()
    }
}

impl MenuNav {
    pub fn new() -> Self {
        MenuNav::Chooser {
            focus: ChooserItem::Host,
        }
    }

    fn lobby(hosting: bool) -> Self {
        if hosting {
            MenuNav::HostLobby {
                focus: LobbyItem::Start,
            }
        } else {
            MenuNav::JoinLobby
        }
    }

    pub fn step(&mut self, input: MenuInput, lobby_len: usize) -> MenuAction {
        match self {
            MenuNav::Chooser { focus } => match input {
                MenuInput::Up => {
                    *focus = match focus {
                        ChooserItem::Host => ChooserItem::Quit,
                        ChooserItem::Solo => ChooserItem::Host,
                        ChooserItem::Join => ChooserItem::Solo,
                        ChooserItem::Quit => ChooserItem::Join,
                    };
                    MenuAction::None
                }
                MenuInput::Down => {
                    *focus = match focus {
                        ChooserItem::Host => ChooserItem::Solo,
                        ChooserItem::Solo => ChooserItem::Join,
                        ChooserItem::Join => ChooserItem::Quit,
                        ChooserItem::Quit => ChooserItem::Host,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => match focus {
                    ChooserItem::Host => {
                        *self = MenuNav::lobby(true);
                        MenuAction::Host
                    }
                    ChooserItem::Solo => {
                        *self = MenuNav::new();
                        MenuAction::StartSolo
                    }
                    ChooserItem::Join => {
                        *self = MenuNav::lobby(false);
                        MenuAction::Join
                    }
                    ChooserItem::Quit => MenuAction::Quit,
                },
                // Console convention: Back at the root highlights Quit rather than
                // exiting outright — B is muscle-memory for leaving nested screens,
                // so a raw B here must not kill the app.
                MenuInput::Back => {
                    *focus = ChooserItem::Quit;
                    MenuAction::None
                }
            },
            MenuNav::HostLobby { focus } => match input {
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        LobbyItem::Start => LobbyItem::Cancel,
                        LobbyItem::Cancel => LobbyItem::Start,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => match focus {
                    LobbyItem::Start => {
                        if lobby_len <= 1 {
                            *self = MenuNav::new();
                            MenuAction::StartSolo
                        } else {
                            MenuAction::StartNetworked
                        }
                    }
                    LobbyItem::Cancel => {
                        *self = MenuNav::new();
                        MenuAction::Cancel
                    }
                },
                MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
            MenuNav::JoinLobby => match input {
                MenuInput::Up | MenuInput::Down => MenuAction::None,
                MenuInput::Confirm | MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
            MenuNav::Disconnected { focus } => match input {
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        DisconnectedItem::Rejoin => DisconnectedItem::Leave,
                        DisconnectedItem::Leave => DisconnectedItem::Rejoin,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => match focus {
                    DisconnectedItem::Rejoin => {
                        *self = MenuNav::Rejoining;
                        MenuAction::Rejoin
                    }
                    DisconnectedItem::Leave => {
                        *self = MenuNav::new();
                        MenuAction::DismissRejoin
                    }
                },
                MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::DismissRejoin
                }
            },
            MenuNav::Rejoining => match input {
                MenuInput::Up | MenuInput::Down => MenuAction::None,
                MenuInput::Confirm | MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
        }
    }

    pub fn focus_chooser(&mut self, item: ChooserItem) {
        if let MenuNav::Chooser { focus } = self {
            *focus = item;
        }
    }

    pub fn focus_lobby(&mut self, item: LobbyItem) {
        if let MenuNav::HostLobby { focus } = self {
            *focus = item;
        }
    }

    pub fn disconnected() -> Self {
        MenuNav::Disconnected {
            focus: DisconnectedItem::Rejoin,
        }
    }

    pub fn focus_disconnected(&mut self, item: DisconnectedItem) {
        if let MenuNav::Disconnected { focus } = self {
            *focus = item;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SyncStamp;

    /// Poll `f` until it yields, or fail after `secs` — real-iroh lobby formation involves
    /// endpoint binding, discovery, and the membership barrier, all wall-clock.
    fn wait_for<T>(secs: u64, what: &str, mut f: impl FnMut() -> Option<T>) -> T {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        loop {
            if let Some(v) = f() {
                return v;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out after {secs}s waiting for {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// Share the host's code, join by it, and wait for both rosters to see 2 peers —
    /// the lobby state every scenario below starts from.
    fn two_peer_lobby() -> (Formation, Formation) {
        let mut host = begin(&StartChoice::Host, 7, None, SyncStamp::ZERO);
        assert!(host.hosting, "Host formation is flagged hosting");
        // The bind is pollable (rl#412): the join code appears once the session lands —
        // on native, the first poll.
        let code = wait_for(15, "the host session to bind", || {
            match host.poll() {
                None => (),
                Some(Ok(_)) => panic!("host formation resolved while binding"),
                Some(Err(e)) => panic!("host bind failed: {e:#}"),
            }
            host.display_code()
        });

        let mut join = begin(&StartChoice::Join(Some(code)), 7, None, SyncStamp::ZERO);
        assert!(!join.hosting, "Join formation is not hosting");
        assert_eq!(
            join.display_code(),
            Some(code),
            "the joiner's lobby shows the code it is dialing"
        );

        // Resolving (either way) before the host presses Start is itself a failure.
        wait_for(30, "both rosters to reach 2 peers", || {
            for (f, who) in [(&mut host, "host"), (&mut join, "joiner")] {
                match f.poll() {
                    None => (),
                    Some(Ok(_)) => panic!("{who} formation resolved before Start"),
                    Some(Err(e)) => panic!("{who} formation failed: {e:#}"),
                }
            }
            (host.roster().len() == 2 && join.roster().len() == 2).then_some(())
        });
        (host, join)
    }

    /// The full 2-peer lobby flow the egui menu drives, over real iroh on this box —
    /// exactly the objects `render::menu` holds: share the host's code, join by it,
    /// both rosters fill, only the host's Start forms the match (rl#94 liveness).
    #[test]
    #[ignore = "binds real iroh UDP endpoints via begin(); run explicitly with --ignored"]
    fn two_peer_lobby_forms_one_match_on_host_start() {
        let _serial = crate::real_net_serial();
        let (mut host, mut join) = two_peer_lobby();

        for (f, who) in [(&host, "host"), (&join, "joiner")] {
            let id = f.my_id().expect("bound at begin");
            assert!(
                f.roster().contains(&id),
                "the {who} finds itself in its roster (the \"(you)\" tag)"
            );
        }

        // A joiner's Start press must arm nothing — the structural guard lives in the
        // core (`Membership::set_starting` no-ops outside host mode, unit-tested there);
        // this presses it end-to-end (an immediate poll() couldn't falsify it alone —
        // the barrier takes wall-clock time).
        join.request_start();
        assert!(
            host.poll().is_none() && join.poll().is_none(),
            "nobody forms before the HOST presses Start"
        );

        host.request_start();
        // Resolve BOTH formations before touching either result: unwrapping one drops its
        // NetDriver (endpoint and all), and the other peer may still be mid-barrier.
        let mut results = (None, None);
        wait_for(30, "both formations after Start", || {
            if results.0.is_none() {
                results.0 = host.poll();
            }
            if results.1.is_none() {
                results.1 = join.poll();
            }
            (results.0.is_some() && results.1.is_some()).then_some(())
        });
        let sim_of = |r: Option<Result<MatchResult>>, who: &str| -> ClientSim {
            match r.unwrap().expect(who) {
                MatchResult::Joined(joined) => joined.0,
                MatchResult::Alone => panic!("{who}: fell back to solo with a peer in the lobby"),
                MatchResult::Cancelled => panic!("{who}: cancelled without a cancel"),
            }
        };
        let h = sim_of(results.0, "host forms");
        let j = sim_of(results.1, "joiner forms");
        assert_eq!(h.peers(), j.peers(), "one match: identical rosters");
        assert_eq!(h.peers().len(), 2);
        assert_ne!(h.me(), j.me(), "distinct player ids");
    }

    /// Cancel from either role resolves that peer's formation as Cancelled (no round) —
    /// which `ready_from` maps back to the chooser.
    #[test]
    #[ignore = "binds real iroh UDP endpoints via begin(); run explicitly with --ignored"]
    fn cancel_leaves_the_lobby_cleanly() {
        let _serial = crate::real_net_serial();
        let (mut host, mut join) = two_peer_lobby();

        join.cancel();
        let result = wait_for(15, "the joiner's formation to resolve", || join.poll());
        assert!(
            matches!(
                result.expect("cancel is not an error"),
                MatchResult::Cancelled
            ),
            "a cancelled joiner resolves Cancelled"
        );
        assert!(
            ready_from(MatchResult::Cancelled, 7).is_none(),
            "…which arms no round"
        );

        host.cancel();
        let result = wait_for(15, "the host's formation to resolve", || host.poll());
        assert!(
            matches!(
                result.expect("cancel is not an error"),
                MatchResult::Cancelled
            ),
            "a cancelled host resolves Cancelled"
        );
    }

    #[test]
    fn alone_becomes_a_solo_round() {
        let seed = 0xABCD;
        let m = ready_from(MatchResult::Alone, seed).expect("Alone is a playable solo round");
        assert!(m.net.is_none(), "Alone is offline — no NetDriver");
        assert_eq!(m.client.me().0, 0, "solo player is id 0");
    }

    #[test]
    fn cancelled_is_not_a_round() {
        assert!(
            ready_from(MatchResult::Cancelled, 0).is_none(),
            "a cancelled lobby yields no round"
        );
    }

    #[test]
    fn chooser_cycles_host_solo_join_quit() {
        let mut nav = MenuNav::new();
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Host
            }
        );
        for expected in [
            ChooserItem::Solo,
            ChooserItem::Join,
            ChooserItem::Quit,
            ChooserItem::Host,
        ] {
            assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
            assert_eq!(nav, MenuNav::Chooser { focus: expected });
        }
        for expected in [
            ChooserItem::Quit,
            ChooserItem::Join,
            ChooserItem::Solo,
            ChooserItem::Host,
        ] {
            assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
            assert_eq!(nav, MenuNav::Chooser { focus: expected });
        }
    }

    #[test]
    fn chooser_solo_starts_a_round_with_no_lobby() {
        // Down once from boot: Solo. Confirm arms straight away — no lobby screen,
        // no session — and the nav resets for the post-round menu return.
        let mut nav = MenuNav::new();
        nav.step(MenuInput::Down, 0);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::StartSolo);
        assert_eq!(nav, MenuNav::new());
    }

    #[test]
    fn chooser_quit_is_one_press_away_and_leaves_nav_intact() {
        // Up from boot wraps straight to Quit (rl#263).
        let mut nav = MenuNav::new();
        assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Quit
            }
        );
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Quit);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Quit
            },
            "Quit doesn't change screens — the app is exiting"
        );

        // Down past Solo and Join reaches it too.
        let mut nav = MenuNav::new();
        nav.step(MenuInput::Down, 0);
        nav.step(MenuInput::Down, 0);
        nav.step(MenuInput::Down, 0);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Quit);

        // Back highlights Quit but never exits by itself — quitting stays two presses.
        let mut nav = MenuNav::new();
        assert_eq!(nav.step(MenuInput::Back, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Quit
            }
        );
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Quit);
    }

    #[test]
    fn confirm_host_or_join_enters_the_lobby_with_the_right_role() {
        let mut host = MenuNav::new();
        assert_eq!(host.step(MenuInput::Confirm, 0), MenuAction::Host);
        assert_eq!(
            host,
            MenuNav::HostLobby {
                focus: LobbyItem::Start
            }
        );

        let mut join = MenuNav::new();
        join.step(MenuInput::Down, 0); // Solo
        join.step(MenuInput::Down, 0); // Join
        assert_eq!(join.step(MenuInput::Confirm, 0), MenuAction::Join);
        assert_eq!(join, MenuNav::JoinLobby);
    }

    #[test]
    fn host_start_resolves_solo_vs_networked_by_roster() {
        let mut alone = MenuNav::lobby(true);
        assert_eq!(alone.step(MenuInput::Confirm, 1), MenuAction::StartSolo);
        assert_eq!(alone, MenuNav::new(), "solo Start resets to the chooser");

        let mut empty = MenuNav::lobby(true);
        assert_eq!(empty.step(MenuInput::Confirm, 0), MenuAction::StartSolo);

        let mut networked = MenuNav::lobby(true);
        assert_eq!(
            networked.step(MenuInput::Confirm, 2),
            MenuAction::StartNetworked
        );
        assert_eq!(
            networked,
            MenuNav::HostLobby {
                focus: LobbyItem::Start
            },
            "networked Start stays in the lobby until the match forms"
        );
    }

    #[test]
    fn host_lobby_navigates_and_cancels() {
        let mut nav = MenuNav::lobby(true);
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::HostLobby {
                focus: LobbyItem::Cancel
            }
        );
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Cancel);
        assert_eq!(nav, MenuNav::new(), "Cancel returns to the chooser");

        let mut back = MenuNav::lobby(true);
        assert_eq!(back.step(MenuInput::Back, 5), MenuAction::Cancel);
        assert_eq!(back, MenuNav::new());
    }

    #[test]
    fn joiner_lobby_can_only_cancel() {
        let mut nav = MenuNav::lobby(false);
        assert_eq!(nav, MenuNav::JoinLobby);
        assert_eq!(nav.step(MenuInput::Down, 9), MenuAction::None);
        assert_eq!(nav, MenuNav::JoinLobby);
        assert_eq!(nav.step(MenuInput::Confirm, 9), MenuAction::Cancel);
        assert_eq!(nav, MenuNav::new());
    }

    #[test]
    fn disconnected_prompt_rejoins_or_leaves() {
        let mut nav = MenuNav::disconnected();
        assert_eq!(
            nav,
            MenuNav::Disconnected {
                focus: DisconnectedItem::Rejoin
            },
            "the prompt starts on the affirmative"
        );
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Disconnected {
                focus: DisconnectedItem::Leave
            }
        );
        assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Rejoin);
        assert_eq!(nav, MenuNav::Rejoining);

        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(nav, MenuNav::Rejoining);
        assert_eq!(nav.step(MenuInput::Back, 0), MenuAction::Cancel);
        assert_eq!(
            nav,
            MenuNav::new(),
            "an abandoned rejoin lands on the chooser"
        );

        let mut decline = MenuNav::disconnected();
        decline.step(MenuInput::Down, 0);
        assert_eq!(
            decline.step(MenuInput::Confirm, 0),
            MenuAction::DismissRejoin
        );
        assert_eq!(decline, MenuNav::new());

        let mut backed = MenuNav::disconnected();
        assert_eq!(backed.step(MenuInput::Back, 0), MenuAction::DismissRejoin);
        assert_eq!(backed, MenuNav::new());
    }

    #[test]
    fn click_focuses_then_confirms_like_a_controller() {
        let mut nav = MenuNav::new();
        nav.focus_chooser(ChooserItem::Join);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Join);

        let mut chooser = MenuNav::new();
        chooser.focus_lobby(LobbyItem::Cancel);
        assert_eq!(
            chooser,
            MenuNav::new(),
            "focus_lobby is inert on the chooser"
        );
    }
}
