/// Right rail: board, breadcrumb, action buttons, overall mix at this node.
///
/// "Overall mix" is the weighted average over all live combos at this node —
/// tells you what the solver does with the *whole range* in this spot, which
/// the per-cell grid can't easily summarize.
library;

import 'package:flutter/material.dart';

import '../../ffi/actions.dart';
import '../../widgets/card_view.dart';
import '../action_palette.dart';
import '../scenario.dart';
import '../training_state.dart';

class NodePanel extends StatelessWidget {
  final TrainingState state;
  const NodePanel({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    final s = state.scenario;
    final n = state.currentNode;
    if (s == null || n == null) {
      return const _Placeholder();
    }
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF324E7A), width: 1),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _scenarioHeader(s),
            const SizedBox(height: 14),
            _board(n),
            const SizedBox(height: 14),
            _potStack(n, s),
            const SizedBox(height: 8),
            _rangeSurvival(s, n),
            const SizedBox(height: 14),
            _breadcrumb(state),
            const SizedBox(height: 14),
            _actionList(state, n),
            const SizedBox(height: 14),
            if (n.isAction) _overallMix(n),
          ],
        ),
      ),
    );
  }

  Widget _scenarioHeader(Scenario s) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          s.label,
          style: const TextStyle(
            color: Color(0xFFEAE6D9),
            fontWeight: FontWeight.w800,
            fontSize: 14,
          ),
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 2),
        Text(
          'pot ${s.startingPot} · stack ${s.effectiveStack} · '
          'expl ${s.exploitability.toStringAsFixed(2)}',
          style: const TextStyle(
            color: Color(0x88EAE6D9),
            fontFamily: 'monospace',
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  Widget _board(ScenarioNode n) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final c in n.board)
          CardView(card: EngineCard.parse(c), width: 60),
      ],
    );
  }

  Widget _potStack(ScenarioNode n, Scenario s) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF34383B),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        children: [
          _kv('POT', '${n.pot}'),
          const SizedBox(width: 14),
          // OOP/IP is the data label (matches the JSON schema); BB/SB pairs
          // it with the positional name HU players think in. Postflop in
          // HU NLHE: BB = OOP, SB = IP.
          _kv('OOP/BB', '${n.stacks[0]}'),
          const SizedBox(width: 10),
          _kv('IP/SB', '${n.stacks[1]}'),
          if (n.isAction && n.player != null) ...[
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: n.player == 'oop'
                    ? const Color(0xFF1F3A56)
                    : const Color(0xFF564423),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '${n.player == 'oop' ? 'OOP/BB' : 'IP/SB'} to act',
                style: TextStyle(
                  color: n.player == 'oop'
                      ? const Color(0xFF6FB3DC)
                      : const Color(0xFFDCBE6F),
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.6,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _rangeSurvival(Scenario s, ScenarioNode n) {
    final oopMass = s.oopMassAt(n);
    final ipMass = s.ipMassAt(n);
    if (oopMass == null || ipMass == null) return const SizedBox.shrink();
    final oopPct = s.oopRootMass > 0 ? oopMass / s.oopRootMass : 0.0;
    final ipPct = s.ipRootMass > 0 ? ipMass / s.ipRootMass : 0.0;
    return Row(
      children: [
        const Text(
          'Range left:',
          style: TextStyle(
            color: Color(0x88EAE6D9),
            fontSize: 11,
            letterSpacing: 0.6,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(width: 8),
        _pctChip('OOP/BB', oopPct, const Color(0xFF6FB3DC)),
        const SizedBox(width: 6),
        _pctChip('IP/SB', ipPct, const Color(0xFFDCBE6F)),
      ],
    );
  }

  Widget _pctChip(String who, double pct, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        border: Border.all(color: color.withValues(alpha: 0.5), width: 1),
        borderRadius: BorderRadius.circular(3),
      ),
      child: Text(
        '$who ${(pct * 100).toStringAsFixed(0)}%',
        style: TextStyle(
          color: color,
          fontFamily: 'monospace',
          fontSize: 11,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$k:',
          style: const TextStyle(
            color: Color(0x88EAE6D9),
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.6,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          v,
          style: const TextStyle(
            color: Color(0xFFEAE6D9),
            fontFamily: 'monospace',
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  Widget _breadcrumb(TrainingState state) {
    final crumbs = state.breadcrumb();
    final atRoot = state.currentId.isEmpty;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Quick-nav strip: Back ascends one action, Root jumps to the top.
        // Both are also reachable by clicking breadcrumb entries below, but
        // having an explicit pair makes "go back and try a different branch"
        // the obvious workflow on this screen.
        Row(
          children: [
            _NavButton(
              icon: Icons.arrow_back,
              label: 'Back',
              enabled: !atRoot,
              onTap: state.up,
            ),
            const SizedBox(width: 6),
            _NavButton(
              icon: Icons.first_page,
              label: 'Root',
              enabled: !atRoot,
              onTap: state.backToRoot,
            ),
          ],
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 4,
          runSpacing: 4,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            for (int i = 0; i < crumbs.length; i++) ...[
              if (i > 0)
                const Text(
                  '›',
                  style: TextStyle(
                    color: Color(0x55EAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 13,
                  ),
                ),
              InkWell(
                onTap: () => state.jumpTo(crumbs[i].id),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: i == crumbs.length - 1
                        ? const Color(0xFF1B2D40)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(3),
                  ),
                  child: Text(
                    crumbs[i].label,
                    style: TextStyle(
                      color: i == crumbs.length - 1
                          ? const Color(0xFFEAE6D9)
                          : const Color(0xCCEAE6D9),
                      fontFamily: 'monospace',
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ],
    );
  }

  Widget _actionList(TrainingState state, ScenarioNode n) {
    if (n.isTerminal) {
      return const Text(
        'Terminal — hand has ended (fold or showdown).',
        style: TextStyle(
          color: Color(0x99EAE6D9),
          fontStyle: FontStyle.italic,
          fontSize: 12,
        ),
      );
    }
    if (n.isChance) {
      return Text(
        'Chance node — pick a card from the runout grid in the center to '
        'descend into that turn/river continuation.\n\n'
        '${n.children.length} cards reachable.',
        style: const TextStyle(
          color: Color(0x99EAE6D9),
          fontSize: 12,
          height: 1.4,
        ),
      );
    }
    if (n.isChancePending) {
      return const Text(
        'Chance node — turn or river deals next.\n'
        '(Re-solve at "Flop + Turn" depth to explore turn cards.)',
        style: TextStyle(
          color: Color(0x99EAE6D9),
          fontStyle: FontStyle.italic,
          fontSize: 12,
          height: 1.4,
        ),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'CONTINUE',
          style: TextStyle(
            color: Color(0x88EAE6D9),
            fontSize: 11,
            letterSpacing: 1.6,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        for (final c in n.children)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: _ActionButton(
              edge: c,
              style: styleFor(c.action, n.pot),
              onTap: () => state.descend(c),
            ),
          ),
      ],
    );
  }

  Widget _overallMix(ScenarioNode n) {
    // Weighted average over combos: freq[a] = Σ w[h]·strat[h][a] / Σ w[h]
    final num = List<double>.filled(n.actions.length, 0.0);
    double tot = 0;
    for (int h = 0; h < n.weights.length; h++) {
      final w = n.weights[h];
      if (w <= 0) continue;
      tot += w;
      for (int a = 0; a < n.actions.length; a++) {
        num[a] += w * n.strategy[h][a];
      }
    }
    if (tot <= 0) return const SizedBox.shrink();
    final freqs = [for (final v in num) v / tot];
    final styles = n.actions.map((a) => styleFor(a, n.pot)).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'OVERALL MIX',
          style: TextStyle(
            color: Color(0x88EAE6D9),
            fontSize: 11,
            letterSpacing: 1.6,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        // One stacked bar.
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: SizedBox(
            height: 18,
            child: Row(
              children: [
                for (int a = 0; a < freqs.length; a++)
                  if (freqs[a] > 0.001)
                    Expanded(
                      flex: (freqs[a] * 1000).round(),
                      child: Container(color: styles[a].color),
                    ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 6),
        for (int a = 0; a < freqs.length; a++)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 1),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: styles[a].color,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    styles[a].longLabel,
                    style: const TextStyle(
                      color: Color(0xCCEAE6D9),
                      fontSize: 12,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Text(
                  '${(freqs[a] * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    color: Color(0xFFEAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final ChildEdge edge;
  final ActionStyle style;
  final VoidCallback onTap;

  const _ActionButton({
    required this.edge,
    required this.style,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: style.color.withValues(alpha: 0.18),
          border: Border.all(color: style.color, width: 1.4),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: style.color,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                edge.action,
                style: const TextStyle(
                  color: Color(0xFFEAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            const Icon(
              Icons.chevron_right,
              size: 18,
              color: Color(0x99EAE6D9),
            ),
          ],
        ),
      ),
    );
  }
}

/// Compact navigation chip used for "Back" and "Root" above the breadcrumb.
/// Kept private to this file because its sizing and palette are tuned to fit
/// the right rail; the rest of the app uses Material's OutlinedButton.
class _NavButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool enabled;
  final VoidCallback onTap;

  const _NavButton({
    required this.icon,
    required this.label,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final fg = enabled
        ? const Color(0xFFEAE6D9)
        : const Color(0x55EAE6D9);
    final border = enabled
        ? const Color(0xFF6FB3DC)
        : const Color(0x336FB3DC);
    return InkWell(
      onTap: enabled ? onTap : null,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          border: Border.all(color: border, width: 1),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: fg),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                color: fg,
                fontFamily: 'monospace',
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Placeholder extends StatelessWidget {
  const _Placeholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
      ),
      child: const Center(
        child: Text(
          'Pick a scenario from the left rail.',
          style: TextStyle(
            color: Color(0x99EAE6D9),
            fontStyle: FontStyle.italic,
            fontSize: 13,
          ),
        ),
      ),
    );
  }
}
