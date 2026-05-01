/// Widget tests for the inspector. The full app needs libpokertrainer.so
/// loaded, which is awkward in `flutter test` (no controlled .so path), so
/// most live-engine coverage lives in tool/ffi_smoke.dart instead.
library;

import 'package:flutter_test/flutter_test.dart';

import 'package:pokertrainer_ui/ffi/actions.dart';

void main() {
  test('Card encoding round-trips through engine convention', () {
    expect(EngineCard.parse('As'), equals(EngineCard.make(12, 3)));
    expect(EngineCard.parse('2c'), equals(EngineCard.make(0, 0)));
    expect(EngineCard.toReadable(EngineCard.parse('Th')), equals('Th'));
    expect(EngineCard.toReadable(EngineCard.noCard), equals('--'));
  });

  test('ActionType ordinals match the engine enum', () {
    expect(ActionType.fold.index,      0);
    expect(ActionType.checkCall.index, 1);
    expect(ActionType.raise25.index,   2);
    expect(ActionType.allIn.index,     10);
    expect(ActionType.numActions,      11);
  });

  test('Player and Street ordinals match', () {
    expect(Player.sb.index, 0);
    expect(Player.bb.index, 1);
    expect(Street.preflop.index,  0);
    expect(Street.river.index,    3);
    expect(Street.showdown.index, 4);
    expect(Street.flop.nVisibleBoard, 3);
    expect(Street.turn.nVisibleBoard, 4);
  });
}
