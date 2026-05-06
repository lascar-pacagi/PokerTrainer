/// Left rail: list of scenarios discovered on disk. Click an entry to load.
library;

import 'package:flutter/material.dart';

import '../scenario_library.dart';
import 'lateral_tooltip.dart';

class ScenarioPicker extends StatelessWidget {
  final ScenarioLibrary library;
  final void Function(ScenarioEntry) onSelect;
  final String? selectedPath;

  const ScenarioPicker({
    super.key,
    required this.library,
    required this.onSelect,
    this.selectedPath,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF272A2D),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF4A4E52), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 8, 4),
            child: Row(
              children: [
                const Text(
                  'SCENARIOS',
                  style: TextStyle(
                    color: Color(0xCCEAE6D9),
                    fontSize: 12,
                    letterSpacing: 1.6,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const Spacer(),
                IconButton(
                  tooltip: 'Rescan disk',
                  iconSize: 16,
                  padding: EdgeInsets.zero,
                  visualDensity: VisualDensity.compact,
                  onPressed: library.rescan,
                  icon: const Icon(Icons.refresh, color: Color(0xAAEAE6D9)),
                ),
              ],
            ),
          ),
          const Divider(height: 1, color: Color(0xFF4A4E52)),
          Expanded(
            child: library.entries.isEmpty
                ? const _EmptyState()
                : ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemCount: library.entries.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 1),
                    itemBuilder: (ctx, i) {
                      final e = library.entries[i];
                      final selected = e.path == selectedPath;
                      final subgame = e.isSubgame;
                      // Tooltip pops to the LEFT of the row (falls back to
                      // the right when the rail is flush with the screen
                      // edge and there's no room left).
                      return LateralTooltip(
                        message: e.label,
                        child: InkWell(
                          onTap: () => onSelect(e),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 10,
                              vertical: 8,
                            ),
                            color: selected
                                ? const Color(0xFF1B2D40)
                                : Colors.transparent,
                            child: Row(
                              children: [
                                Icon(
                                  subgame
                                      ? Icons.alt_route
                                      : Icons.account_tree_outlined,
                                  size: 14,
                                  color: subgame
                                      ? const Color(0xFFD4B43F)
                                      : const Color(0xFF6FB3DC),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    e.label,
                                    style: TextStyle(
                                      color: selected
                                          ? const Color(0xFFEAE6D9)
                                          : const Color(0xCCEAE6D9),
                                      fontFamily: 'monospace',
                                      fontSize: 13,
                                      fontWeight: selected
                                          ? FontWeight.w700
                                          : FontWeight.w400,
                                    ),
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(14),
      child: Text(
        'No scenarios open.\n\n'
        'Click "Open…" to load a saved\n'
        'solve, or "New solve…" to run\n'
        'pt-solver from a spot config.\n\n'
        'The ⟳ button rescans the\n'
        'validation_runs/scenarios/\n'
        'directory if you want to\n'
        'see everything on disk.',
        style: TextStyle(
          color: Color(0x88EAE6D9),
          fontFamily: 'monospace',
          fontSize: 11,
          height: 1.5,
        ),
      ),
    );
  }
}
