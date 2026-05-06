/// Curated preflop range presets — labeled handles you can drop into the
/// OOP/IP range fields of the new-solve dialog. These are *approximations*
/// of common GTO-ish ranges drawn from public charts and rounded to clean
/// hand-class boundaries; they're meant as starting points for postflop
/// study, not authoritative preflop solves.
///
/// All ranges are 100 bb effective, no antes, standard NLHE.
///
/// Categories:
///   * **HU** — heads-up. SB acts first preflop and is the in-position player
///     postflop (HU only). BB defends out of position.
///   * **6-max** — six-handed cash. Useful when you want to study postflop
///     spots that arise from a normal full-ring opening (e.g. BTN-open / BB
///     defend → broad calling range out of position).
library;

class PreflopPreset {
  final String label;
  final String range;
  final String description;
  /// Group used to organize the dropdown (e.g. "Heads-Up", "6-max BTN").
  final String group;

  const PreflopPreset({
    required this.label,
    required this.range,
    required this.description,
    required this.group,
  });
}

/// All presets, in display order. The dropdown groups by `group`.
const List<PreflopPreset> preflopPresets = [
  // ── Heads-Up 100bb ──────────────────────────────────────────────────
  PreflopPreset(
    label: 'HU SB Open (Min-raise)',
    group: 'Heads-Up 100bb',
    description: 'SB raises ~80% of hands HU. Folds only the worst offsuit.',
    range:
        '22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,73s+,63s+,53s+,43s,'
        'A2o+,K2o+,Q4o+,J6o+,T7o+,97o+,87o,76o',
  ),
  PreflopPreset(
    label: 'HU BB Defend (vs SB minraise)',
    group: 'Heads-Up 100bb',
    description: 'BB call+3-bet range vs SB open. ~70% of hands continue.',
    range:
        '22+,A2s+,K2s+,Q2s+,J3s+,T5s+,95s+,85s+,75s+,64s+,54s,43s,'
        'A2o+,K5o+,Q8o+,J9o+,T9o',
  ),
  PreflopPreset(
    label: 'HU BB 3-Bet (vs SB minraise)',
    group: 'Heads-Up 100bb',
    description: 'BB 3-bet polar mix: value + bluffs.',
    range: 'JJ+,AKo,AQs+,A5s-A2s,KQs,KJs,QJs,T9s,87s,76s,65s',
  ),
  PreflopPreset(
    label: 'HU SB Call vs BB 3-Bet',
    group: 'Heads-Up 100bb',
    description: "SB's calling range when BB 3-bets. Tighter than its open.",
    range:
        '22-TT,AQs-A2s,K2s+,Q5s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,'
        'AJo,KQo,KJo,QJo',
  ),

  // ── 6-max 100bb opens ────────────────────────────────────────────────
  PreflopPreset(
    label: 'BTN Open',
    group: '6-max Opens',
    description: '~52% of hands. Standard 6-max button-open range.',
    range:
        '22+,A2s+,K5s+,Q7s+,J7s+,T7s+,97s+,86s+,75s+,64s+,54s,43s,'
        'A8o+,KTo+,QTo+,JTo,T9o,98o,87o',
  ),
  PreflopPreset(
    label: 'CO Open',
    group: '6-max Opens',
    description: '~28% of hands. Cutoff opening range.',
    range:
        '22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,86s+,75s+,65s,54s,'
        'ATo+,KJo+,QJo,JTo',
  ),
  PreflopPreset(
    label: 'MP Open',
    group: '6-max Opens',
    description: '~18% of hands. Middle position / Lojack.',
    range: '22+,ATs+,KTs+,QTs+,J9s+,T9s,98s,87s,76s,65s,AJo+,KQo',
  ),
  PreflopPreset(
    label: 'UTG Open',
    group: '6-max Opens',
    description: '~14% of hands. Tight UTG opening range.',
    range: '22+,ATs+,KJs+,QJs,JTs,T9s,98s,87s,76s,AJo+,KQo',
  ),

  // ── 6-max defends ────────────────────────────────────────────────────
  PreflopPreset(
    label: 'BB vs BTN Open — Call',
    group: '6-max Defends',
    description: 'BB flat range vs BTN open. Pretty wide for postflop play.',
    range:
        '22-JJ,A2s-AJs,K2s+,Q4s+,J6s+,T6s+,95s+,84s+,74s+,63s+,53s+,42s+,32s,'
        'A2o-AJo,K5o+,Q8o+,J8o+,T8o+,97o+,86o+,75o+,65o,54o',
  ),
  PreflopPreset(
    label: 'BB vs BTN Open — 3-Bet',
    group: '6-max Defends',
    description: 'Polar 3-bet range vs BTN: pairs+ for value, suited bluffs.',
    range: 'QQ+,AKo,AKs,AQs,A5s-A2s,K5s,Q5s,J8s,T8s,97s,86s,75s',
  ),
  PreflopPreset(
    label: 'BB vs UTG Open — Call',
    group: '6-max Defends',
    description: 'Tighter defend vs UTG. Mostly speculative & connected hands.',
    range: '22-99,A9s-A2s,K9s+,Q9s+,J9s+,T8s+,98s,87s,76s,65s,54s,KJo+,QJo',
  ),
  PreflopPreset(
    label: 'BB vs UTG Open — 3-Bet',
    group: '6-max Defends',
    description: 'Tight 3-bet range vs UTG (premiums + select bluffs).',
    range: 'TT+,AKs,AKo,AQs,A5s,A4s,KQs',
  ),

  // ── Generic ─────────────────────────────────────────────────────────
  PreflopPreset(
    label: 'Top 25% (any position)',
    group: 'Generic',
    description: 'Roughly the top quarter of hands. Decent default range.',
    range: '22+,A2s+,K9s+,Q9s+,JTs,T9s,98s,87s,76s,65s,A9o+,KTo+,QJo',
  ),
  PreflopPreset(
    label: 'Top 10%',
    group: 'Generic',
    description: 'Tight range — premium pairs and high suited connectors.',
    range: '77+,ATs+,KTs+,QTs+,JTs,AJo+,KQo',
  ),

  // ── PokerCoaching 100bb HU charts ──────────────────────────────────────
  // Visual transcriptions of the PokerCoaching "100bb HUNL Cash Game" chart
  // PDF (Jonathan Little). Sizing assumptions (per the chart instructions):
  // 2.5bb open, 4× 3-bet, 2.2× 4-bet, 5-bet shoves. Cells are weighted —
  // the editor renders the mixed strategies; load one as a preset and
  // tweak from there.
  //
  // Tools: see ui/tool/build_pokercoaching_ranges.dart for the source
  // matrices and the round-trip from grid → string.
  PreflopPreset(
    label: 'BTN RFI (81%)',
    group: 'PokerCoaching HU 100bb',
    description: 'Button raise-first-in. Very wide — folds only the bottom-'
        'left offsuit junk (Q4o-T4o, Q3o-63o, Q2o-32o etc.).',
    range:
        '22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,'
        'A2o+,K2o+,Q5o+,J5o+,T5o+,95o+,85o+,74o+,64o+,53o+,43o',
  ),
  PreflopPreset(
    label: 'BB 3-Bet vs BTN open (20.4%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BB 3-bet response to BTN/SB 2.5bb open. Polar mix: '
        'value with premiums, mid-frequency bluffs with suited Broadways and '
        'connectors.',
    range:
        '22:0.2,77-44:0.5,88+,A9s-A2s:0.5,ATs+,K4s-K2s:0.2,K6s-K5s:0.35,'
        'K8s-K7s:0.5,K9s:0.75,KTs+,Q4s:0.2,Q6s-Q5s:0.35,Q8s-Q7s:0.5,'
        'Q9s:0.75,QTs+,J5s:0.2,J6s:0.35,J8s-J7s:0.5,J9s:0.75,JTs,'
        'T7s-T6s:0.2,T8s:0.5,T9s,95s:0.2,96s:0.5,97s:0.75,98s,84s:0.2,'
        '85s:0.5,86s:0.75,87s,73s:0.2,74s:0.5,75s:0.75,76s,63s:0.5,64s:0.75,'
        '65s,53s:0.75,54s,42s:0.2,43s:0.5,32s:0.2,A9o-A8o:0.1,ATo:0.2,'
        'AJo:0.5,AQo+,KTo-K8o:0.1,KJo:0.2,KQo:0.5,QTo-Q8o:0.1,QJo:0.2,'
        'J8o+:0.1,T8o+:0.1,98o:0.1,87o:0.1,65o:0.1,54o:0.1',
  ),
  PreflopPreset(
    label: 'BB Call vs BTN open (55.22%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BB flat-call response to BTN/SB 2.5bb open. Wide range — '
        'every speculative hand that doesn\'t belong in the 3-bet bucket. '
        'Note premiums (top-left) are 0 here; they go in the 3-bet range.',
    range:
        '22:0.8,77-44:0.5,A9s-A2s:0.5,K4s-K2s:0.8,K6s-K5s:0.65,K8s-K7s:0.5,'
        'K9s:0.25,Q4s:0.8,Q6s-Q5s:0.65,Q8s-Q7s:0.5,Q9s:0.25,J4s-J2s,'
        'J5s:0.8,J7s-J6s:0.65,J8s:0.5,J9s:0.25,T5s-T2s,T7s-T6s:0.8,T8s:0.5,'
        '94s-92s,95s:0.8,96s:0.5,97s:0.25,84s-82s,85s:0.8,86s:0.5,87s:0.25,'
        '73s-72s,74s:0.8,75s:0.5,76s:0.25,62s,63s:0.8,64s:0.5,65s:0.25,'
        '53s-52s,54s:0.25,42s+:0.8,32s:0.8,A7o-A2o,A9o-A8o:0.9,ATo:0.8,'
        'AJo:0.5,K4o-K2o:0.9,K7o-K5o,KTo-K8o:0.9,KJo:0.8,KQo:0.5,'
        'Q4o-Q3o:0.5,Q5o:0.75,Q7o-Q6o,QTo-Q8o:0.9,QJo:0.8,J5o:0.5,J7o-J6o,'
        'J8o+:0.9,T5o:0.35,T7o-T6o,T8o+:0.9,95o:0.35,97o-96o,98o:0.9,'
        '85o:0.35,86o,87o:0.9,74o:0.5,75o,76o:0.9,64o+:0.9,53o:0.25,'
        '54o:0.5,43o:0.25',
  ),
  PreflopPreset(
    label: 'BTN 4-Bet vs BB 3-bet (5.1%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BTN 4-bet response to BB 3-bet. Tight value-heavy with '
        'low-frequency bluffs. KK+/AKo/AQs are pure; the rest are mixed.',
    range:
        '88:0.1,99:0.2,KK-TT,A9s-A2s:0.02,AJs-ATs:0.1,AQs:0.5,K3s-K2s:0.04,'
        'K9s-K4s:0.07,KTs+:0.02,Q5s-Q2s:0.04,Q9s-Q6s:0.07,QTs+:0.02,'
        'J5s-J2s:0.04,J9s-J6s:0.07,JTs:0.02,T5s-T2s:0.04,T6s+:0.07,'
        '94s-92s:0.04,97s-95s:0.07,98s:0.04,83s-82s:0.04,86s-84s:0.07,'
        '87s:0.1,72s:0.04,75s-73s:0.07,76s:0.1,64s-63s:0.07,65s:0.1,'
        '52s:0.04,53s:0.07,54s:0.1,43s:0.1,32s:0.04,A3o-A2o:0.15,'
        'A5o-A4o:0.2,A9o-A6o:0.1,AJo-ATo:0.12,AQo:0.15,AKo,KTo-K6o:0.04,'
        'KJo:0.05,KQo:0.1,Q9o-Q6o:0.04,QTo+:0.05,J9o-J7o:0.04,JTo:0.05,'
        'T7o+:0.04,97o+:0.04,87o:0.04,76o:0.04',
  ),
  PreflopPreset(
    label: 'BTN Call vs BB 3-bet (37.5%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BTN flatting range vs BB 3-bet. Hands that are too good to '
        'fold but not strong enough to 4-bet for value. Mid-strength value '
        'and a bunch of suited speculation.',
    range:
        '77-22,88:0.9,99:0.8,ATs-A2s:0.98,AQs-AJs:0.96,AKs:0.5,K9s-K2s:0.96,'
        'KTs+:0.98,Q3s-Q2s:0.5,Q9s-Q4s:0.96,QTs+:0.98,J4s-J3s:0.5,'
        'J9s-J5s:0.96,JTs:0.98,T4s:0.5,T5s+:0.96,94s:0.5,95s+:0.96,'
        '86s-84s:0.96,87s:0.93,73s:0.5,75s-74s:0.96,76s:0.93,62s:0.5,'
        '63s+:0.96,52s:0.5,53s:0.96,54s:0.93,42s:0.5,43s:0.93,32s:0.5,'
        'A3o-A2o:0.2,A5o-A4o:0.6,A7o-A6o:0.2,A9o-A8o:0.75,AJo-ATo:0.88,'
        'AQo:0.85,K8o:0.1,K9o:0.6,KJo-KTo:0.95,KQo:0.9,Q8o:0.1,Q9o:0.6,'
        'QTo+:0.95,J8o:0.1,J9o:0.6,JTo:0.95,T8o-T7o:0.1,T9o:0.6,98o:0.1,'
        '65o:0.1',
  ),
  PreflopPreset(
    label: 'BB 5-Bet vs BTN 4-bet (3.7%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BB 5-bet (all-in) range. Mostly the absolute nuts (KK+, '
        'AKs, AKo) plus a tiny mixed-frequency bluff component.',
    range:
        'TT:0.5,KK-JJ,AA:0.65,A3s-A2s:0.2,A8s-A7s:0.2,AKs,98s:0.2,ATo:0.05,'
        'AJo:0.1,AKo,KQo:0.05',
  ),
  PreflopPreset(
    label: 'BB Call vs BTN 4-bet (13.42%)',
    group: 'PokerCoaching HU 100bb',
    description: 'BB flat-call response to BTN 4-bet. Hands too marginal to '
        '5-bet shove but worth seeing a flop in a deep, capped pot.',
    range:
        '33-22:0.2,44:0.5,AA:0.35,A3s-A2s:0.5,A5s-A4s,A7s-A6s:0.25,A8s:0.5,'
        'AQs-A9s,K8s:0.5,K9s:0.75,KTs+,Q8s:0.75,Q9s+,J7s:0.75,J8s+,T7s:0.75,'
        'T8s+,96s:0.75,97s,98s:0.8,86s+,75s+,64s+,53s+,43s:0.5,ATo:0.1,'
        'AJo:0.5,AQo,KJo:0.1,KQo:0.5,QJo:0.1,87o:0.5',
  ),
];

/// Index by group for the dropdown.
Map<String, List<PreflopPreset>> presetsByGroup() {
  final map = <String, List<PreflopPreset>>{};
  for (final p in preflopPresets) {
    map.putIfAbsent(p.group, () => []).add(p);
  }
  return map;
}
