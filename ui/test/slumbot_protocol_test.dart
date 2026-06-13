import 'package:flutter_test/flutter_test.dart';
import 'package:pokertrainer_ui/ffi/actions.dart';
import 'package:pokertrainer_ui/slumbot/slumbot_protocol.dart';

void main() {
  group('parseActionString', () {
    test('empty → no tokens', () {
      expect(parseActionString(''), isEmpty);
    });

    test('multi-street betting line', () {
      final t = parseActionString('b200c/kk/b300c');
      expect(t.length, 6);
      // preflop: bet-to-200, call
      expect(t[0].street, 0);
      expect((t[0].action as SbBet).toChips, 200);
      expect(t[1].action, isA<SbCall>());
      // flop: check, check
      expect(t[2].street, 1);
      expect(t[2].action, isA<SbCheck>());
      expect(t[3].action, isA<SbCheck>());
      // turn: bet-to-300, call
      expect(t[4].street, 2);
      expect((t[4].action as SbBet).toChips, 300);
      expect(t[5].action, isA<SbCall>());
    });

    test('fold + all-in size', () {
      final t = parseActionString('b20000f');
      expect((t[0].action as SbBet).toChips, 20000);
      expect(t[1].action, isA<SbFold>());
    });

    test('malformed token throws', () {
      expect(() => parseActionString('x'), throwsFormatException);
      expect(() => parseActionString('b'), throwsFormatException); // size-less bet
    });
  });

  group('encodeAction', () {
    test('fold', () {
      expect(encodeAction(ActionType.fold, toCallChips: 100), 'f');
    });
    test('check vs call', () {
      expect(encodeAction(ActionType.checkCall, toCallChips: 0), 'k');
      expect(encodeAction(ActionType.checkCall, toCallChips: 100), 'c');
    });
    test('raise / all-in carry the chip target', () {
      expect(encodeAction(ActionType.raise75, toCallChips: 100, betToChips: 350), 'b350');
      expect(encodeAction(ActionType.allIn, toCallChips: 100, betToChips: 20000), 'b20000');
    });
  });

  group('cards', () {
    test('parseCards round-trips through EngineCard', () {
      final bytes = parseCards(['Ah', 'Kd']);
      expect(EngineCard.toReadable(bytes[0]), 'Ah');
      expect(EngineCard.toReadable(bytes[1]), 'Kd');
    });
    test('boardSlots pads unrevealed slots with NO_CARD', () {
      final slots = boardSlots(['2c', '7h', 'Ts']);
      expect(slots.length, 5);
      expect(EngineCard.toReadable(slots[0]), '2c');
      expect(EngineCard.toReadable(slots[2]), 'Ts');
      expect(slots[3], EngineCard.noCard);
      expect(slots[4], EngineCard.noCard);
    });
  });
}
