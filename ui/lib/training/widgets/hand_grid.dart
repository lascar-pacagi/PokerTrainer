/// 13×13 strategy chart: each cell is colored by the mixed-strategy frequency
/// for the player to act, action segments stacked left-to-right.
///
/// Tap a cell to see the per-combo breakdown (TODO v2 — for now just shows
/// the cell label + per-action %s in a tooltip on hover).
library;

import 'package:flutter/material.dart';

import '../action_palette.dart';
import '../hand_class.dart';
import '../scenario.dart';

class HandGrid extends StatelessWidget {
  final ScenarioNode node;
  final List<String> combos;
  /// Tap callback for non-empty cells. Receives the cell that was tapped;
  /// host wires this to the combo-detail dialog.
  final void Function(ChartCell)? onCellTap;

  const HandGrid({
    super.key,
    required this.node,
    required this.combos,
    this.onCellTap,
  });

  @override
  Widget build(BuildContext context) {
    if (!node.isAction) {
      return Center(
        child: Text(
          node.isTerminal
              ? 'Terminal node — hand has ended.'
              : 'Chance node — turn/river deals next.',
          style: const TextStyle(
            color: Color(0x99EAE6D9),
            fontStyle: FontStyle.italic,
            fontSize: 14,
          ),
        ),
      );
    }
    final grid = buildChart(
      combos: combos,
      strategy: node.strategy,
      ev: node.ev,
      weights: node.weights,
      numActions: node.actions.length,
    );
    final styles = node.actions.map((a) => styleFor(a, node.pot)).toList();

    return LayoutBuilder(builder: (context, constraints) {
      const headerSize = 26.0;
      final available = constraints.biggest.shortestSide - headerSize - 8;
      // Upper clamp 72 (was 48): on wider windows the chart now scales up to
      // a comfortable size instead of leaving big margins around itself.
      final cellSize = (available / 13).clamp(28.0, 72.0).toDouble();
      final gridSize = cellSize * 13 + headerSize;

      return Center(
        child: SizedBox(
          width: gridSize,
          height: gridSize,
          child: Column(
            children: [
              _columnHeader(headerSize, cellSize),
              Expanded(
                child: Row(
                  children: [
                    _rowHeader(headerSize, cellSize),
                    Expanded(
                      child: _gridBody(grid, styles, cellSize),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
    });
  }

  Widget _columnHeader(double headerSize, double cellSize) {
    return SizedBox(
      height: headerSize,
      child: Row(
        children: [
          SizedBox(width: headerSize),
          for (final r in chartRanks)
            SizedBox(
              width: cellSize,
              child: Center(
                child: Text(
                  r,
                  style: const TextStyle(
                    color: Color(0x99EAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _rowHeader(double headerSize, double cellSize) {
    return SizedBox(
      width: headerSize,
      child: Column(
        children: [
          for (final r in chartRanks)
            SizedBox(
              height: cellSize,
              child: Center(
                child: Text(
                  r,
                  style: const TextStyle(
                    color: Color(0x99EAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _gridBody(
    List<List<ChartCell?>> grid,
    List<ActionStyle> styles,
    double cellSize,
  ) {
    return Column(
      children: [
        for (int r = 0; r < 13; r++)
          Expanded(
            child: Row(
              children: [
                for (int c = 0; c < 13; c++)
                  Expanded(
                    child: _GridCell(
                      cell: grid[r][c],
                      styles: styles,
                      size: cellSize,
                      onTap: onCellTap,
                    ),
                  ),
              ],
            ),
          ),
      ],
    );
  }
}

class _GridCell extends StatelessWidget {
  final ChartCell? cell;
  final List<ActionStyle> styles;
  final double size;
  final void Function(ChartCell)? onTap;

  const _GridCell({
    required this.cell,
    required this.styles,
    required this.size,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final empty = cell == null;
    final tooltip = empty
        ? 'No combos'
        : _tooltipText(cell!, styles);
    final tap = (!empty && onTap != null) ? () => onTap!(cell!) : null;

    return Tooltip(
      message: tooltip,
      waitDuration: const Duration(milliseconds: 250),
      child: Padding(
        padding: const EdgeInsets.all(0.5),
        child: Material(
          color: empty ? const Color(0xFF2D3134) : const Color(0xFF3D4145),
          borderRadius: BorderRadius.circular(2),
          child: InkWell(
            onTap: tap,
            borderRadius: BorderRadius.circular(2),
            child: empty
                ? const SizedBox.shrink()
                : _filledCell(cell!),
          ),
        ),
      ),
    );
  }

  Widget _filledCell(ChartCell cell) {
    return Stack(
      children: [
        // Stacked horizontal bars — proportional to action freqs.
        Positioned.fill(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: Row(
              children: [
                for (int a = 0; a < cell.actionFreqs.length; a++)
                  if (cell.actionFreqs[a] > 0.001)
                    Expanded(
                      flex: (cell.actionFreqs[a] * 1000).round(),
                      child: Container(color: styles[a].color),
                    ),
              ],
            ),
          ),
        ),
        Positioned.fill(
          child: Center(
            child: Text(
              cell.label,
              style: TextStyle(
                color: Colors.white,
                fontSize: size > 56 ? 16 : (size > 44 ? 14 : (size > 34 ? 12 : 11)),
                fontFamily: 'monospace',
                fontWeight: FontWeight.w800,
                shadows: const [
                  Shadow(
                    color: Color(0xCC000000),
                    blurRadius: 2,
                    offset: Offset(0, 1),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  String _tooltipText(ChartCell cell, List<ActionStyle> styles) {
    final buf = StringBuffer();
    buf.writeln('${cell.label}  (weight ${cell.totalWeight.toStringAsFixed(2)})');
    for (int a = 0; a < cell.actionFreqs.length; a++) {
      final f = cell.actionFreqs[a];
      if (f < 0.001) continue;
      buf.writeln(
        '  ${styles[a].longLabel}: ${(f * 100).toStringAsFixed(1)}%  '
        'EV ${cell.actionEvs[a].toStringAsFixed(1)}',
      );
    }
    return buf.toString().trimRight();
  }
}
