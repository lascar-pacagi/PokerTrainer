// JSON-bridge CLI for postflop-solver.
//
// Reads one spot description from --input (or stdin), runs the solver to a
// target exploitability (or iteration cap), writes the root strategy of the
// to-act side as JSON to --output (or stdout).
//
// The Python driver (trainer/validation/solver_distance.py) shells out to this
// binary once per sampled spot and compares the GTO action distribution to the
// model's action distribution on the same hole cards.

use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use postflop_solver::{
    card_from_str, card_to_string, flop_from_str, holes_to_strings, solve, ActionTree,
    BetSizeOptions, BoardState, CardConfig, PostFlopGame, Range, TreeConfig, NOT_DEALT,
};

// ─── JSON contract ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Serialize)]
struct PlayerStrategy {
    private_cards: Vec<String>,
    /// Shape: num_hands × num_actions. Rows sum to 1 (within float tolerance).
    strategy: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct SpotOutput {
    exploitability: f32,
    /// Actions formatted as in `available_actions()` debug-print, e.g.
    /// "Check", "Bet(100)", "AllIn(900)", "Fold", "Call", "Raise(300)".
    actions: Vec<String>,
    /// The to-act side at the root. For BoardState::Flop with no prior action,
    /// this is OOP. We hard-code OOP for v1; downstream code will need to
    /// extend this once we sample non-root spots.
    oop: PlayerStrategy,
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "pt-solver", about = "JSON-bridge for postflop-solver")]
struct Cli {
    /// Path to input JSON. Reads stdin if omitted.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Path to output JSON. Writes stdout if omitted.
    #[arg(long)]
    output: Option<PathBuf>,
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

// ─── Solver invocation ──────────────────────────────────────────────────────

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

fn solve_spot(spec: &SpotInput) -> Result<SpotOutput> {
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
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| anyhow!("PostFlopGame: {}", e))?;

    let (mem, mem_compressed) = game.memory_usage();
    eprintln!(
        "pt-solver: tree built. memory: {:.2} GB (32-bit) / {:.2} GB (compressed)",
        mem as f64 / 1e9,
        mem_compressed as f64 / 1e9,
    );
    // Refuse to allocate if it would exceed ~8 GB; surface a clear error
    // instead of getting SIGKILL'd by the OOM-killer.
    const MAX_BYTES: u64 = 8_000_000_000;
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
    game.cache_normalized_weights();

    // Root strategy of the to-act side. At the root of a Flop tree with no
    // prior action this is OOP (player 0).
    let actions: Vec<String> = game
        .available_actions()
        .iter()
        .map(|a| format!("{:?}", a))
        .collect();
    let strat_flat = game.strategy();
    let private_cards = game.private_cards(0).to_vec();
    let n_hands = private_cards.len();
    let n_actions = actions.len();
    debug_assert_eq!(strat_flat.len(), n_hands * n_actions);

    // Crate layout: strategy[hand + action * n_hands]. Transpose to per-hand rows.
    let mut by_hand = Vec::with_capacity(n_hands);
    for h in 0..n_hands {
        let mut row = Vec::with_capacity(n_actions);
        for a in 0..n_actions {
            row.push(strat_flat[h + a * n_hands]);
        }
        by_hand.push(row);
    }

    let private_strs = holes_to_strings(&private_cards)
        .map_err(|e| anyhow!("holes_to_strings: {}", e))?;

    Ok(SpotOutput {
        exploitability,
        actions,
        oop: PlayerStrategy {
            private_cards: private_strs,
            strategy: by_hand,
        },
    })
}

fn main() -> Result<()> {
    // Quiet the stderr: silence unused warnings for card_to_string in v1.
    let _ = card_to_string;

    let cli = Cli::parse();
    let raw = read_input(cli.input.as_ref())?;
    let spec: SpotInput = serde_json::from_str(&raw).context("parsing input JSON")?;
    let out = solve_spot(&spec)?;
    let body = serde_json::to_string_pretty(&out).context("serializing output JSON")?;
    write_output(cli.output.as_ref(), &body)?;
    Ok(())
}
