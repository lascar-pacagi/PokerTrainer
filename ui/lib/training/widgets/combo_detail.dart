/// Modal dialog showing every combo inside one 13×13 hand-class cell, with
/// per-combo strategy, weight, and EV.
///
/// Lets the user check whether the solver treats e.g. AhKs differently from
/// AdKc — important for blocker analysis. The cell tooltip shows aggregates;
/// this view shows per-combo detail.
library;

import 'package:flutter/material.dart';

import '../action_palette.dart';
import '../hand_class.dart';
import '../scenario.dart';

class ComboDetailDialog extends StatelessWidget {
  final ScenarioNode node;
  final List<String> combos;
  final ChartCell cell;

  const ComboDetailDialog({
    super.key,
    required this.node,
    required this.combos,
    required this.cell,
  });

  @override
  Widget build(BuildContext context) {
    final styles = node.actions.map((a) => styleFor(a, node.pot)).toList();
    final rows = <_Row>[];
    for (int h = 0; h < combos.length; h++) {
      final c = combos[h];
      try {
        final cls = classifyCombo(c);
        if (cls.row != cell.row || cls.col != cell.col) continue;
      } catch (_) {
        continue;
      }
      final w = h < node.weights.length ? node.weights[h] : 0.0;
      if (w <= 0.0) continue;
      rows.add(_Row(
        combo: c,
        weight: w,
        strategy: node.strategy[h],
        ev: node.ev[h],
      ));
    }
    rows.sort((a, b) => b.weight.compareTo(a.weight));

    return Dialog(
      backgroundColor: const Color(0xFF272A2D),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10),
        side: const BorderSide(color: Color(0xFF324E7A), width: 1.4),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 640, maxHeight: 720),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Text(
                    cell.label,
                    style: const TextStyle(
                      color: Color(0xFFEAE6D9),
                      fontFamily: 'monospace',
                      fontSize: 28,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Text(
                    '${rows.length} combo${rows.length == 1 ? "" : "s"} · '
                    'weight ${cell.totalWeight.toStringAsFixed(2)}',
                    style: const TextStyle(
                      color: Color(0x99EAE6D9),
                      fontSize: 15,
                      fontFamily: 'monospace',
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.close, size: 20),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              _legend(styles),
              const SizedBox(height: 10),
              const Divider(height: 1, color: Color(0xFF4A4E52)),
              Expanded(
                child: ListView.separated(
                  itemCount: rows.length,
                  separatorBuilder: (_, _) =>
                      const Divider(height: 1, color: Color(0xFF2D3134)),
                  itemBuilder: (ctx, i) => _comboRow(rows[i], styles),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _legend(List<ActionStyle> styles) {
    return Wrap(
      spacing: 14,
      runSpacing: 4,
      children: [
        for (final st in styles)
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 13,
                height: 13,
                decoration: BoxDecoration(
                  color: st.color,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 6),
              Text(
                st.longLabel,
                style: const TextStyle(
                  color: Color(0xCCEAE6D9),
                  fontSize: 14,
                ),
              ),
            ],
          ),
      ],
    );
  }

  Widget _comboRow(_Row r, List<ActionStyle> styles) {
    // Tooltip recaps the per-combo data with extra precision (3 decimals on
    // strategy + EV) so users can verify near-equilibrium ties that round
    // identically at the inline 2-decimal display.
    final tip = StringBuffer();
    tip.writeln('${r.combo}  ·  weight ${r.weight.toStringAsFixed(3)}');
    for (int a = 0; a < r.strategy.length; a++) {
      final f = r.strategy[a];
      if (f < 0.001) continue;
      tip.writeln(
        '  ${styles[a].longLabel}: ${(f * 100).toStringAsFixed(2)}%  '
        'EV ${r.ev[a].toStringAsFixed(3)}',
      );
    }
    return Tooltip(
      message: tip.toString().trimRight(),
      waitDuration: const Duration(milliseconds: 350),
      child: Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              SizedBox(
                width: 70,
                child: Text(
                  r.combo,
                  style: const TextStyle(
                    color: Color(0xFFEAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Text(
                'w ${r.weight.toStringAsFixed(2)}',
                style: const TextStyle(
                  color: Color(0x88EAE6D9),
                  fontFamily: 'monospace',
                  fontSize: 13,
                ),
              ),
              const Spacer(),
              for (int a = 0; a < r.strategy.length; a++)
                if (r.strategy[a] > 0.005)
                  Padding(
                    padding: const EdgeInsets.only(left: 10),
                    child: Text(
                      '${styles[a].shortLabel} ${(r.strategy[a] * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        color: styles[a].color,
                        fontFamily: 'monospace',
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: SizedBox(
              height: 10,
              child: Row(
                children: [
                  for (int a = 0; a < r.strategy.length; a++)
                    if (r.strategy[a] > 0.001)
                      Expanded(
                        flex: (r.strategy[a] * 1000).round(),
                        child: Container(color: styles[a].color),
                      ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              const Text(
                'EV: ',
                style: TextStyle(
                  color: Color(0x66EAE6D9),
                  fontSize: 13,
                  fontFamily: 'monospace',
                ),
              ),
              for (int a = 0; a < r.ev.length; a++)
                if (r.strategy[a] > 0.005)
                  Padding(
                    padding: const EdgeInsets.only(right: 12),
                    // 2 decimals so equilibrium ties don't read as gaps
                    // due to rounding (e.g. 10.451 vs 10.453 displays as
                    // "10.45 = 10.45" rather than "10.5 ≠ 10.4").
                    child: Text(
                      '${styles[a].shortLabel}=${r.ev[a].toStringAsFixed(2)}',
                      style: const TextStyle(
                        color: Color(0xAAEAE6D9),
                        fontFamily: 'monospace',
                        fontSize: 13,
                      ),
                    ),
                  ),
            ],
          ),
        ],
      ),
      ),
    );
  }
}

class _Row {
  final String combo;
  final double weight;
  final List<double> strategy;
  final List<double> ev;

  const _Row({
    required this.combo,
    required this.weight,
    required this.strategy,
    required this.ev,
  });
}
