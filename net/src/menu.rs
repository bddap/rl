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

    pub fn display_code(&self) -> Option<EndpointId> {
        if self.hosting {
            if self.bound.get().is_none()
                && let Ok(id) = self.bound_rx.try_recv()
            {
                self.bound.set(Some(id));
            }
            self.bound.get()
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
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        ChooserItem::Host => ChooserItem::Join,
                        ChooserItem::Join => ChooserItem::Host,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => {
                    let hosting = matches!(focus, ChooserItem::Host);
                    *self = MenuNav::lobby(hosting);
                    if hosting {
                        MenuAction::Host
                    } else {
                        MenuAction::Join
                    }
                }
                MenuInput::Back => MenuAction::None,
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

    #[test]
    #[ignore = "binds a real iroh UDP endpoint via begin(); run explicitly with --ignored"]
    fn only_host_holds_the_start_command() {
        let host = begin(&StartChoice::Host, 0, None, 0, 1);
        assert!(host.hosting, "Host formation is flagged hosting");
        host.request_start();
        host.cancel();

        let join = begin(&StartChoice::Join(None), 0, None, 0, 1);
        assert!(!join.hosting, "Join formation is not hosting");
        join.request_start();
        join.cancel();
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
        assert_eq!(nav.step(MenuInput::Back, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
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
