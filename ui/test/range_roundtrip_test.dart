import 'package:flutter_test/flutter_test.dart';
import 'package:pokertrainer_ui/training/range_parser.dart';
import 'package:pokertrainer_ui/training/preflop_presets.dart';

double pct(Map<String, double> w) {
  var combos = 0.0;
  w.forEach((cls, weight) {
    final n = cls.length == 2 ? 6 : (cls.endsWith('s') ? 4 : 12);
    combos += n * weight;
  });
  return 100.0 * combos / 1326.0;
}

void main() {
  test('round-trip parse and serialize', () {
    final samples = [
      '66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs',
      'JJ+,AKo,AQs+,A5s-A2s,KQs,KJs,QJs,T9s,87s,76s,65s',
      '22+',
    ];
    for (final s in samples) {
      final parsed = parseRangeString(s);
      final back = serializeClassWeights(parsed);
      // Re-parse the serialized version; it should yield the same map.
      final reparsed = parseRangeString(back);
      expect(reparsed.length, parsed.length,
          reason: 'class count drift on "$s" → "$back"');
      // Combo-coverage percentage should also match.
      expect(pct(reparsed), closeTo(pct(parsed), 0.01),
          reason: 'pct drift on "$s"');
    }
  });
  test('combo-level inputs fold into class', () {
    final p = parseRangeString('AhKh:0.5,AdKd:0.5');
    expect(p['AKs'], closeTo(0.5, 0.01));
  });
  test('all presets parse to non-empty', () {
    for (final preset in preflopPresets) {
      final p = parseRangeString(preset.range);
      expect(p.isNotEmpty, true, reason: 'Preset ${preset.label} parsed empty');
    }
  });
}
