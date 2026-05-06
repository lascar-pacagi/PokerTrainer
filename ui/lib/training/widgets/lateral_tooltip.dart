/// Tooltip that pops out *to the side* of its trigger widget rather than
/// above/below. Built on `MouseRegion` + `OverlayEntry` because Flutter's
/// built-in `Tooltip` only supports vertical positioning.
///
/// Prefers the left side; falls back to the right when there's not enough
/// room on the left (e.g. when the trigger sits in a left-edge rail).
library;

import 'dart:async';

import 'package:flutter/material.dart';

class LateralTooltip extends StatefulWidget {
  final String message;
  final Widget child;
  final Duration waitDuration;
  final double maxWidth;

  const LateralTooltip({
    super.key,
    required this.message,
    required this.child,
    this.waitDuration = const Duration(milliseconds: 350),
    this.maxWidth = 360,
  });

  @override
  State<LateralTooltip> createState() => _LateralTooltipState();
}

class _LateralTooltipState extends State<LateralTooltip> {
  OverlayEntry? _entry;
  Timer? _showTimer;

  @override
  void dispose() {
    _showTimer?.cancel();
    _hide();
    super.dispose();
  }

  void _scheduleShow() {
    _showTimer?.cancel();
    _showTimer = Timer(widget.waitDuration, _show);
  }

  void _show() {
    if (!mounted || _entry != null) return;
    final renderBox = context.findRenderObject() as RenderBox?;
    if (renderBox == null) return;
    final pos = renderBox.localToGlobal(Offset.zero);
    final size = renderBox.size;
    final screenW = MediaQuery.of(context).size.width;

    // Prefer left of the trigger; fall back to right if there's not enough
    // room (typical when the trigger lives in a left-edge rail).
    const gap = 8.0;
    final preferLeft = pos.dx >= widget.maxWidth + gap;
    final overlay = Overlay.of(context);

    _entry = OverlayEntry(
      builder: (ctx) {
        // Cap the tooltip's horizontal span on whichever side we picked.
        final available = preferLeft
            ? pos.dx - gap
            : screenW - (pos.dx + size.width + gap);
        final width = available.clamp(120.0, widget.maxWidth);
        return Positioned(
          left: preferLeft
              ? pos.dx - gap - width
              : pos.dx + size.width + gap,
          top: pos.dy,
          width: width,
          child: IgnorePointer(
            child: Material(
              color: Colors.transparent,
              child: Container(
                decoration: BoxDecoration(
                  color: const Color(0xFF1B1F22),
                  border: Border.all(color: const Color(0xFF6FB3DC), width: 1),
                  borderRadius: BorderRadius.circular(4),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x66000000),
                      blurRadius: 8,
                      offset: Offset(0, 2),
                    ),
                  ],
                ),
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                child: Text(
                  widget.message,
                  softWrap: true,
                  style: const TextStyle(
                    color: Color(0xFFEAE6D9),
                    fontFamily: 'monospace',
                    fontSize: 13,
                    height: 1.4,
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
    overlay.insert(_entry!);
  }

  void _hide() {
    _entry?.remove();
    _entry = null;
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => _scheduleShow(),
      onExit: (_) {
        _showTimer?.cancel();
        _hide();
      },
      child: widget.child,
    );
  }
}
