use std::sync::mpsc;
use std::thread;

use anyhow::Result;
pub use iroh::EndpointId;

use crate::client::ClientSim;
use crate::formation::{self, LobbyControl};
use crate::membership::Role;
use crate::net_loop::{self, MatchResult, NetDriver};

#[derive(Debug, Clone)]
pub enum StartChoice {
    Host,
    Join(Option<EndpointId>),
}

const NET_EXPECT: usize = 2;

pub struct Formation {
    rx: mpsc::Receiver<Result<MatchResult>>,
    dial_code: Option<EndpointId>,
    bound_rx: mpsc::Receiver<EndpointId>,
    bound: std::cell::Cell<Option<EndpointId>>,
    start_tx: std::cell::Cell<Option<mpsc::Sender<()>>>,
    cancel_tx: mpsc::Sender<()>,
    roster_rx: mpsc::Receiver<Vec<EndpointId>>,
    roster: std::cell::RefCell<Vec<EndpointId>>,
    pub hosting: bool,
}

impl Formation {
    pub fn poll(&self) -> Option<Result<MatchResult>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("formation thread ended unexpectedly")))
            }
        }
    }

    /// This peer's own endpoint id, once its session has bound — available in either
    /// role (the wire sends it unconditionally).
    pub fn my_id(&self) -> Option<EndpointId> {
        if self.bound.get().is_none()
            && let Ok(id) = self.bound_rx.try_recv()
        {
            self.bound.set(Some(id));
        }
        self.bound.get()
    }

    pub fn display_code(&self) -> Option<EndpointId> {
        if self.hosting {
            self.my_id()
        } else {
            self.dial_code
        }
    }

    pub fn roster(&self) -> Vec<EndpointId> {
        while let Ok(r) = self.roster_rx.try_recv() {
            *self.roster.borrow_mut() = r;
        }
        self.roster.borrow().clone()
    }

    pub fn lobby_len(&self) -> usize {
        self.roster().len()
    }

    pub fn request_start(&self) {
        if let Some(tx) = self.start_tx.take() {
            let _ = tx.send(());
        }
    }

    pub fn cancel(&self) {
        let _ = self.cancel_tx.send(());
    }
}

fn spawn_formation(
    seed: u64,
    join: Option<EndpointId>,
    hosting: bool,
    telemetry: Option<EndpointId>,
    asset_digest: u64,
    crab_count: u8,
) -> Formation {
    let (tx, rx) = mpsc::channel();
    let (bound_tx, bound_rx) = mpsc::channel();
    let (roster_tx, roster_rx) = mpsc::channel();
    let (start_tx, start_rx) = mpsc::channel();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let (role, host_start_tx) = if hosting {
        (Role::Host, Some(start_tx))
    } else {
        (Role::Joiner, None)
    };
    thread::spawn(move || {
        let result = net_loop::connect_and_form_lobby(
            seed,
            NET_EXPECT,
            join,
            telemetry,
            Some(bound_tx),
            LobbyControl {
                role,
                start_rx,
                cancel_rx,
                roster_tx,
            },
            asset_digest,
            crab_count,
        );
        let _ = tx.send(result);
    });
    Formation {
        rx,
        dial_code: join,
        bound_rx,
        bound: std::cell::Cell::new(None),
        start_tx: std::cell::Cell::new(host_start_tx),
        cancel_tx,
        roster_rx,
        roster: std::cell::RefCell::new(Vec::new()),
        hosting,
    }
}

pub fn begin(
    choice: &StartChoice,
    seed: u64,
    telemetry: Option<EndpointId>,
    asset_digest: u64,
    crab_count: u8,
) -> Formation {
    match choice {
        StartChoice::Host => spawn_formation(seed, None, true, telemetry, asset_digest, crab_count),
        StartChoice::Join(host) => {
            spawn_formation(seed, *host, false, telemetry, asset_digest, crab_count)
        }
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
                        ChooserItem::Join => ChooserItem::Host,
                        ChooserItem::Quit => ChooserItem::Join,
                    };
                    MenuAction::None
                }
                MenuInput::Down => {
                    *focus = match focus {
                        ChooserItem::Host => ChooserItem::Join,
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
            thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// Share the host's code, join by it, and wait for both rosters to see 2 peers —
    /// the lobby state every scenario below starts from.
    fn two_peer_lobby() -> (Formation, Formation) {
        let host = begin(&StartChoice::Host, 7, None, 0, 0);
        assert!(host.hosting, "Host formation is flagged hosting");
        let code = wait_for(15, "the host's shareable join code", || host.display_code());

        let join = begin(&StartChoice::Join(Some(code)), 7, None, 0, 0);
        assert!(!join.hosting, "Join formation is not hosting");
        assert_eq!(
            join.display_code(),
            Some(code),
            "the joiner's lobby shows the code it is dialing"
        );

        wait_for(30, "both rosters to reach 2 peers", || {
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
        let (host, join) = two_peer_lobby();

        for (f, who) in [(&host, "host"), (&join, "joiner")] {
            let id = wait_for(15, "this peer's own endpoint id", || f.my_id());
            assert!(
                f.roster().contains(&id),
                "the {who} finds itself in its roster (the \"(you)\" tag)"
            );
        }

        // Only the host holds the start command — a joiner's Start is inert.
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
        let (host, join) = two_peer_lobby();

        join.cancel();
        let result = wait_for(15, "the joiner's formation to resolve", || join.poll());
        assert!(
            matches!(result.expect("cancel is not an error"), MatchResult::Cancelled),
            "a cancelled joiner resolves Cancelled"
        );
        assert!(
            ready_from(MatchResult::Cancelled, 7).is_none(),
            "…which arms no round"
        );

        host.cancel();
        let result = wait_for(15, "the host's formation to resolve", || host.poll());
        assert!(
            matches!(result.expect("cancel is not an error"), MatchResult::Cancelled),
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
    fn chooser_navigates_between_host_and_join() {
        let mut nav = MenuNav::new();
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Host
            }
        );
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
        assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Host
            }
        );
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
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

        // Down past Join reaches it too.
        let mut nav = MenuNav::new();
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
        join.step(MenuInput::Down, 0);
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
