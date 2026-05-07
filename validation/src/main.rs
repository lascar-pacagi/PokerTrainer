// JSON-bridge CLI for postflop-solver.
//
// Two output modes (selected via --mode):
//
//   --mode root    Single-node legacy schema: solver output for the root node
//                  only (OOP first-to-act). Used by trainer/validation/
//                  solver_distance.py for one-shot exploitability checks.
//
//   --mode tree    Full action-tree dump: every action node reachable on the
//                  current street(s) is serialized with its strategy and EV.
//                  Walks stop at chance nodes (turn/river deals); those are
//                  marked `chance_pending` so the UI can render the boundary.
//                  This is what the Flutter training tab consumes.
//
// `tree` is the default. Both modes solve once.

use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use postflop_solver::{
    card_from_str, card_to_string, flop_from_str, holes_to_strings, solve, Action, ActionTree,
    BetSizeOptions, BoardState, CardConfig, PostFlopGame, Range, TreeConfig, NOT_DEALT,
};

// ─── JSON contract: input ────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
struct BetSizesSpec {
    bet: String,
    raise: String,
}

impl BetSizesSpec {
    fn to_options(&self) -> Result<BetSizeOptions> {
        BetSizeOptions::try_from((self.bet.as_str(), self.raise.as_str()))
            .map_err(|e| anyhow!("bet sizes ({:?}, {:?}): {}", self.bet, self.raise, e))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct SpotInput {
    oop_range: String,
    ip_range: String,
    flop: String,
    #[serde(default)]
    turn: Option<String>,
    #[serde(default)]
    river: Option<String>,
    starting_pot: i32,
    effective_stack: i32,
    #[serde(default)]
    rake_rate: f64,
    #[serde(default)]
    rake_cap: f64,
    flop_bet_sizes: BetSizesSpec,
    turn_bet_sizes: BetSizesSpec,
    river_bet_sizes: BetSizesSpec,
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    #[serde(default = "default_target_pct_pot")]
    target_exploitability_pct_pot: f32,
}

fn default_max_iterations() -> u32 {
    1000
}
fn default_target_pct_pot() -> f32 {
    0.5
}

// ─── JSON contract: root-mode output (legacy) ────────────────────────────────

#[derive(Debug, Serialize)]
struct PlayerStrategy {
    private_cards: Vec<String>,
    /// Shape: num_hands × num_actions. Rows sum to 1 (within float tolerance).
    strategy: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct RootOutput {
    exploitability: f32,
    /// Actions formatted as in `available_actions()` debug-print, e.g.
    /// "Check", "Bet(100)", "AllIn(900)", "Fold", "Call", "Raise(300)".
    actions: Vec<String>,
    /// The to-act side at the root. For BoardState::Flop with no prior action,
    /// this is OOP. We hard-code OOP for v1; downstream code will need to
    /// extend this once we sample non-root spots.
    oop: PlayerStrategy,
}

// ─── JSON contract: tree-mode output ─────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PrivateCards {
    oop: Vec<String>,
    ip: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChildEdge {
    /// Index into the parent's `actions` array.
    action_idx: usize,
    /// Same string as the parent's `actions[action_idx]`.
    action: String,
    /// `id` of the child node in the flat `nodes` list.
    node_id: String,
}

/// One entry in the flat `nodes` list. The variant is selected by `kind`:
///
/// * `"action"` — interior decision node. Has `player`, `actions`,
///   `strategy`, `ev`, and `children`.
/// * `"chance_pending"` — turn or river deal. Walk stops here; UI renders
///   "deal next street" boundary.
/// * `"terminal"` — fold or showdown. No further play.
#[derive(Debug, Serialize)]
struct NodeOut {
    /// Stable id derived from history: "" for root, "0" "0/1" "0/1/2" for
    /// children. Slash-separated action indices.
    id: String,
    /// Raw action indices from root to this node.
    history: Vec<usize>,
    /// Human-readable line, e.g. ["Check", "Bet(30)", "Call"].
    line: Vec<String>,
    /// "action" | "chance_pending" | "terminal".
    kind: &'static str,
    /// Cards on the board at this node — flop (3) + optional turn + river.
    board: Vec<String>,
    /// Pot at the time of this decision (chips).
    pot: i32,
    /// Remaining stacks [oop, ip].
    stacks: [i32; 2],

    // ── action-only fields ──────────────────────────────────────────────
    /// "oop" or "ip"; null at chance/terminal.
    #[serde(skip_serializing_if = "Option::is_none")]
    player: Option<&'static str>,
    /// Available actions at this node (matches `children` order).
    #[serde(skip_serializing_if = "Option::is_none")]
    actions: Option<Vec<String>>,
    /// Edges to child nodes. Empty at chance/terminal.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<ChildEdge>,
    /// Strategy of the player to act, indexed by `private_cards.<player>`.
    /// Shape: num_hands × num_actions. Each row sums to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    strategy: Option<Vec<Vec<f32>>>,
    /// Expected value of each action for each hand of the player to act.
    /// Same shape as `strategy`. Units: chips.
    #[serde(skip_serializing_if = "Option::is_none")]
    ev: Option<Vec<Vec<f32>>>,
    /// Reach probabilities for the player to act at this node (length =
    /// num_hands). Use these to weight per-hand strategies when aggregating
    /// into a 13×13 grid; combos that overlap the board have weight 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<Vec<f32>>,
    /// Reach probabilities for the *other* player at this node — handy for
    /// EV/pot-odds aggregations from the receiver's perspective.
    #[serde(skip_serializing_if = "Option::is_none")]
    weights_opp: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct TreeOutput {
    /// Schema version. Bump when fields are renamed/removed.
    version: u32,
    /// Echo of the input spec, for traceability.
    input: SpotInput,
    /// Final exploitability after solve, in chips.
    exploitability: f32,
    /// Convenience copy of pot/stack/board so consumers don't re-parse `input`.
    starting_pot: i32,
    effective_stack: i32,
    flop: [String; 3],
    turn: Option<String>,
    river: Option<String>,
    /// Per-player private hand strings, in the same order the strategy/ev
    /// matrices index. Must be loaded once; node payloads reference by index.
    private_cards: PrivateCards,
    nodes: Vec<NodeOut>,
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Mode {
    /// Single-node legacy output (root strategy only).
    Root,
    /// Full action-tree dump. Default.
    Tree,
}

/// How deep into chance nodes the tree-walker descends.
///
/// * `Flop`     — stop at the first chance (turn deal). Smallest dump.
/// * `Turn`     — walk through every possible turn card and dump that turn's
///                 action subtree, but stop at the river chance.
/// * `River`    — walk through both turn AND river chance levels. Largest
///                 dump; only practical for narrow ranges or shallow stacks.
///
/// Default is `Flop` (matches the previous behaviour).
#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Depth {
    Flop,
    Turn,
    River,
}

#[derive(Parser, Debug)]
#[command(name = "pt-solver", about = "JSON-bridge for postflop-solver")]
struct Cli {
    /// Path to input JSON. Reads stdin if omitted.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Path to output JSON. Writes stdout if omitted.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Output schema. Defaults to `tree` (full action-tree dump).
    #[arg(long, value_enum, default_value_t = Mode::Tree)]
    mode: Mode,
    /// How far the tree-walk descends through chance (turn/river) nodes.
    /// Only relevant in `tree` mode.
    #[arg(long, value_enum, default_value_t = Depth::Flop)]
    depth: Depth,
    /// Build the tree, report memory, then exit without allocating or
    /// solving. Output JSON is written with a small estimate payload (no
    /// strategy data) so the caller can preview tree size cheaply.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

/// Output of `--dry-run`: just the tree size, no strategy.
#[derive(Debug, Serialize)]
struct DryRunOutput {
    /// Schema version — kept aligned with TreeOutput so callers can branch on
    /// `kind: "dry_run"` vs the absence of `nodes`.
    version: u32,
    kind: &'static str,
    /// Bytes the solver would allocate at full precision.
    memory_bytes: u64,
    /// Bytes if `allocate_memory(true)` (compressed) was used.
    memory_bytes_compressed: u64,
    /// True if memory_bytes exceeds the safety cap and a full solve would be
    /// rejected. The UI shows this as a red warning.
    too_large: bool,
    /// Same cap pt-solver enforces in non-dry-run mode.
    memory_cap_bytes: u64,
}

fn read_input(path: Option<&PathBuf>) -> Result<String> {
    match path {
        Some(p) => fs::read_to_string(p).with_context(|| format!("reading {:?}", p)),
        None => {
            let mut s = String::new();
            io::stdin().read_to_string(&mut s).context("reading stdin")?;
            Ok(s)
        }
    }
}

fn write_output(path: Option<&PathBuf>, body: &str) -> Result<()> {
    match path {
        Some(p) => fs::write(p, body).with_context(|| format!("writing {:?}", p)),
        None => {
            io::stdout().write_all(body.as_bytes())?;
            io::stdout().write_all(b"\n")?;
            Ok(())
        }
    }
}

// ─── Solver setup ───────────────────────────────────────────────────────────

fn parse_optional_card(s: Option<&String>) -> Result<u8> {
    match s {
        Some(t) => card_from_str(t).map_err(|e| anyhow!("card_from_str({:?}): {}", t, e)),
        None => Ok(NOT_DEALT),
    }
}

fn initial_state(turn: u8, river: u8) -> BoardState {
    if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    }
}

/// Memory cap enforced before `allocate_memory()`. A full solve at 8 GB is
/// already pushing it; anything larger is almost certainly a config error
/// and we'd rather error fast than get OOM-killed mid-solve.
const MAX_BYTES: u64 = 8_000_000_000;

/// Build the un-allocated game and report its memory needs. Used by both
/// `--dry-run` and full solves; never enforces the cap itself so callers can
/// surface "too large" errors with their own framing.
fn build_game(spec: &SpotInput) -> Result<(PostFlopGame, u64, u64)> {
    let oop: Range = spec
        .oop_range
        .parse()
        .map_err(|e| anyhow!("oop_range parse: {}", e))?;
    let ip: Range = spec
        .ip_range
        .parse()
        .map_err(|e| anyhow!("ip_range parse: {}", e))?;

    let flop = flop_from_str(&spec.flop).map_err(|e| anyhow!("flop_from_str: {}", e))?;
    let turn = parse_optional_card(spec.turn.as_ref())?;
    let river = parse_optional_card(spec.river.as_ref())?;

    let card_config = CardConfig {
        range: [oop, ip],
        flop,
        turn,
        river,
    };

    let flop_bs = spec.flop_bet_sizes.to_options()?;
    let turn_bs = spec.turn_bet_sizes.to_options()?;
    let river_bs = spec.river_bet_sizes.to_options()?;

    let tree_config = TreeConfig {
        initial_state: initial_state(turn, river),
        starting_pot: spec.starting_pot,
        effective_stack: spec.effective_stack,
        rake_rate: spec.rake_rate,
        rake_cap: spec.rake_cap,
        flop_bet_sizes: [flop_bs.clone(), flop_bs],
        turn_bet_sizes: [turn_bs.clone(), turn_bs],
        river_bet_sizes: [river_bs.clone(), river_bs],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).map_err(|e| anyhow!("ActionTree: {}", e))?;
    let game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| anyhow!("PostFlopGame: {}", e))?;

    let (mem, mem_compressed) = game.memory_usage();
    eprintln!(
        "pt-solver: tree built. memory: {:.2} GB (32-bit) / {:.2} GB (compressed)",
        mem as f64 / 1e9,
        mem_compressed as f64 / 1e9,
    );
    Ok((game, mem, mem_compressed))
}

fn build_and_solve(spec: &SpotInput) -> Result<(PostFlopGame, f32)> {
    let (mut game, mem, _) = build_game(spec)?;
    if mem > MAX_BYTES {
        return Err(anyhow!(
            "tree too large for this spot: {:.2} GB > {:.2} GB cap. \
             Reduce bet sizes, lower effective_stack, or tighten add_allin_threshold.",
            mem as f64 / 1e9,
            MAX_BYTES as f64 / 1e9,
        ));
    }
    game.allocate_memory(false);
    let target = (spec.starting_pot as f32) * (spec.target_exploitability_pct_pot / 100.0);
    let exploitability = solve(&mut game, spec.max_iterations, target, false);
    Ok((game, exploitability))
}

// ─── Helpers shared between root- and tree-mode output ──────────────────────

fn action_to_string(a: &Action) -> String {
    // For chance, the user-facing label is the card string ("Th", "2c", ...).
    // The Debug impl would give "Chance(33)" with the raw byte index — useful
    // for debugging but unreadable in a UI.
    if let Action::Chance(card) = a {
        if let Ok(s) = card_to_string(*card) {
            return s;
        }
    }
    // Everything else: "Check", "Bet(30)", "Call", "Raise(120)", "AllIn(970)",
    // "Fold". The Debug impl is the right format.
    format!("{:?}", a)
}

/// Transpose the crate's strategy/EV layout (`[hand + action * n_hands]`) to
/// per-hand rows: `out[hand_idx][action_idx]`.
fn transpose_per_hand(flat: &[f32], n_hands: usize, n_actions: usize) -> Vec<Vec<f32>> {
    debug_assert_eq!(flat.len(), n_hands * n_actions);
    let mut out = Vec::with_capacity(n_hands);
    for h in 0..n_hands {
        let mut row = Vec::with_capacity(n_actions);
        for a in 0..n_actions {
            row.push(flat[h + a * n_hands]);
        }
        out.push(row);
    }
    out
}

fn cards_to_strings(cards: &[u8]) -> Vec<String> {
    cards.iter().map(|&c| card_to_string(c).unwrap_or_else(|_| "?".into())).collect()
}

// ─── Root-mode dump (legacy) ────────────────────────────────────────────────

fn dump_root(game: &mut PostFlopGame, exploitability: f32) -> Result<RootOutput> {
    game.cache_normalized_weights();
    let actions: Vec<String> = game.available_actions().iter().map(action_to_string).collect();
    let private_cards = game.private_cards(0).to_vec();
    let n_hands = private_cards.len();
    let n_actions = actions.len();

    let strat = transpose_per_hand(&game.strategy(), n_hands, n_actions);
    let private_strs = holes_to_strings(&private_cards)
        .map_err(|e| anyhow!("holes_to_strings: {}", e))?;

    Ok(RootOutput {
        exploitability,
        actions,
        oop: PlayerStrategy { private_cards: private_strs, strategy: strat },
    })
}

// ─── Tree-mode dump ─────────────────────────────────────────────────────────

fn history_to_id(h: &[usize]) -> String {
    if h.is_empty() {
        String::new()
    } else {
        h.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("/")
    }
}

/// One entry in the enumeration: the play-history from root (action indices
/// at action nodes; card bytes at chance nodes), the human-readable action
/// labels collected along the way, and how many chance nodes have already
/// been traversed (so `dump_tree` can decide `chance` vs `chance_pending`).
///
/// Lines are built incrementally during the walk so we never re-call
/// `apply_history` just to recover labels.
struct EnumNode {
    history: Vec<usize>,
    line: Vec<String>,
    chance_count: u32,
}

/// Phase 1: enumerate every history we want to emit. Action nodes are always
/// included; chance nodes are walked through up to `max_chance` levels deep
/// (0 = stop at the first chance, 1 = walk turn deals only, 2 = walk turn
/// and river). Terminal nodes are emitted once as leaves.
///
/// **History encoding:** at action nodes the entry is the index into
/// `available_actions()`; at chance nodes it is the dealt card byte (0..52).
/// This matches `PostFlopGame::play`, whose argument means an action index at
/// action nodes but a card byte at chance nodes.
fn enumerate_histories(game: &mut PostFlopGame, max_chance: u32) -> Vec<EnumNode> {
    let mut out: Vec<EnumNode> = Vec::new();
    let mut stack: Vec<EnumNode> = vec![EnumNode {
        history: Vec::new(),
        line: Vec::new(),
        chance_count: 0,
    }];
    while let Some(node) = stack.pop() {
        game.apply_history(&node.history);
        let chance_count = node.chance_count;
        let history = node.history.clone();
        let line = node.line.clone();
        out.push(EnumNode { history: history.clone(), line: line.clone(), chance_count });
        if game.is_terminal_node() {
            continue;
        }
        if game.is_chance_node() {
            if chance_count >= max_chance {
                continue;
            }
            // At chance nodes `play()` takes a card byte. Collect (byte,label)
            // pairs from available_actions(), sort by byte for stable ordering.
            let mut paired: Vec<(usize, String)> = game
                .available_actions()
                .iter()
                .filter_map(|a| match a {
                    Action::Chance(c) => Some((*c as usize, action_to_string(a))),
                    _ => None,
                })
                .collect();
            paired.sort_unstable_by_key(|(c, _)| *c);
            for (c, label) in paired.into_iter().rev() {
                let mut h2 = history.clone();
                let mut l2 = line.clone();
                h2.push(c);
                l2.push(label);
                stack.push(EnumNode {
                    history: h2,
                    line: l2,
                    chance_count: chance_count + 1,
                });
            }
            continue;
        }
        // Plain action node — push action indices.
        let acts: Vec<String> = game.available_actions().iter().map(action_to_string).collect();
        for a in (0..acts.len()).rev() {
            let mut h2 = history.clone();
            let mut l2 = line.clone();
            h2.push(a);
            l2.push(acts[a].clone());
            stack.push(EnumNode { history: h2, line: l2, chance_count });
        }
    }
    out
}

fn dump_tree(
    spec: &SpotInput,
    game: &mut PostFlopGame,
    exploitability: f32,
    max_chance: u32,
) -> Result<TreeOutput> {
    let oop_cards = game.private_cards(0).to_vec();
    let ip_cards = game.private_cards(1).to_vec();
    let private_cards = PrivateCards {
        oop: holes_to_strings(&oop_cards).map_err(|e| anyhow!("holes_to_strings(oop): {}", e))?,
        ip: holes_to_strings(&ip_cards).map_err(|e| anyhow!("holes_to_strings(ip): {}", e))?,
    };
    let n_oop = oop_cards.len();
    let n_ip = ip_cards.len();

    // Phase 1: enumerate histories.
    let enum_nodes = enumerate_histories(game, max_chance);
    eprintln!(
        "pt-solver: tree-mode walk yielded {} nodes (max_chance={})",
        enum_nodes.len(),
        max_chance,
    );

    // Phase 2: visit each, build NodeOut.
    let mut nodes = Vec::with_capacity(enum_nodes.len());
    for en in &enum_nodes {
        let h = &en.history;
        game.apply_history(h);
        let id = history_to_id(h);
        let line = en.line.clone();
        let board = cards_to_strings(&game.current_board());
        let bet = game.total_bet_amount();
        let pot = spec.starting_pot + bet[0] + bet[1];
        let stacks = [spec.effective_stack - bet[0], spec.effective_stack - bet[1]];

        if game.is_terminal_node() {
            nodes.push(NodeOut {
                id,
                history: h.clone(),
                line,
                kind: "terminal",
                board,
                pot,
                stacks,
                player: None,
                actions: None,
                children: Vec::new(),
                strategy: None,
                ev: None,
                weights: None,
                weights_opp: None,
            });
            continue;
        }

        if game.is_chance_node() {
            // Walked? If chance_count < max_chance we descended; emit kind
            // `"chance"` with children. Otherwise stop here as `chance_pending`.
            let walked = en.chance_count < max_chance;
            let (kind, actions, children): (&'static str, Option<Vec<String>>, Vec<ChildEdge>) =
                if walked {
                    // Children are sorted by card byte (matches enumerate_histories).
                    let raw = game.available_actions();
                    let mut paired: Vec<(usize, String)> = raw
                        .iter()
                        .filter_map(|a| match a {
                            Action::Chance(c) => Some((*c as usize, action_to_string(a))),
                            _ => None,
                        })
                        .collect();
                    paired.sort_unstable_by_key(|(c, _)| *c);
                    let acts: Vec<String> = paired.iter().map(|(_, s)| s.clone()).collect();
                    let edges: Vec<ChildEdge> = paired
                        .iter()
                        .map(|(c, label)| {
                            let mut h2 = h.clone();
                            h2.push(*c);
                            ChildEdge {
                                action_idx: *c,
                                action: label.clone(),
                                node_id: history_to_id(&h2),
                            }
                        })
                        .collect();
                    ("chance", Some(acts), edges)
                } else {
                    ("chance_pending", None, Vec::new())
                };
            nodes.push(NodeOut {
                id,
                history: h.clone(),
                line,
                kind,
                board,
                pot,
                stacks,
                player: None,
                actions,
                children,
                strategy: None,
                ev: None,
                weights: None,
                weights_opp: None,
            });
            continue;
        }

        // Action node: read strategy + EV for current player.
        game.cache_normalized_weights();
        let actions: Vec<String> =
            game.available_actions().iter().map(action_to_string).collect();
        let player_idx = game.current_player();
        let player_label = if player_idx == 0 { "oop" } else { "ip" };
        let n_hands = if player_idx == 0 { n_oop } else { n_ip };
        let n_actions = actions.len();

        let strategy = transpose_per_hand(&game.strategy(), n_hands, n_actions);
        let ev = transpose_per_hand(&game.expected_values_detail(player_idx), n_hands, n_actions);
        let weights = game.weights(player_idx).to_vec();
        let weights_opp = game.weights(player_idx ^ 1).to_vec();

        let children: Vec<ChildEdge> = (0..n_actions)
            .map(|a| {
                let mut h2 = h.clone();
                h2.push(a);
                ChildEdge {
                    action_idx: a,
                    action: actions[a].clone(),
                    node_id: history_to_id(&h2),
                }
            })
            .collect();

        nodes.push(NodeOut {
            id,
            history: h.clone(),
            line,
            kind: "action",
            board,
            pot,
            stacks,
            player: Some(player_label),
            actions: Some(actions),
            children,
            strategy: Some(strategy),
            ev: Some(ev),
            weights: Some(weights),
            weights_opp: Some(weights_opp),
        });
    }

    let flop_strs = cards_to_strings(&game.card_config().flop);
    let flop = [flop_strs[0].clone(), flop_strs[1].clone(), flop_strs[2].clone()];
    let turn_str = spec.turn.clone();
    let river_str = spec.river.clone();

    Ok(TreeOutput {
        version: 1,
        input: spec.clone(),
        exploitability,
        starting_pot: spec.starting_pot,
        effective_stack: spec.effective_stack,
        flop,
        turn: turn_str,
        river: river_str,
        private_cards,
        nodes,
    })
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let raw = read_input(cli.input.as_ref())?;
    let spec: SpotInput = serde_json::from_str(&raw).context("parsing input JSON")?;

    if cli.dry_run {
        let (_game, mem, mem_c) = build_game(&spec)?;
        let out = DryRunOutput {
            version: 1,
            kind: "dry_run",
            memory_bytes: mem,
            memory_bytes_compressed: mem_c,
            too_large: mem > MAX_BYTES,
            memory_cap_bytes: MAX_BYTES,
        };
        let body = serde_json::to_string_pretty(&out).context("serializing dry-run output")?;
        write_output(cli.output.as_ref(), &body)?;
        return Ok(());
    }

    let (mut game, exploitability) = build_and_solve(&spec)?;

    let body = match cli.mode {
        Mode::Root => {
            let out = dump_root(&mut game, exploitability)?;
            serde_json::to_string_pretty(&out).context("serializing root output")?
        }
        Mode::Tree => {
            let max_chance = match cli.depth {
                Depth::Flop => 0,
                Depth::Turn => 1,
                Depth::River => 2,
            };
            let out = dump_tree(&spec, &mut game, exploitability, max_chance)?;
            serde_json::to_string(&out).context("serializing tree output")?
        }
    };
    write_output(cli.output.as_ref(), &body)?;
    Ok(())
}
