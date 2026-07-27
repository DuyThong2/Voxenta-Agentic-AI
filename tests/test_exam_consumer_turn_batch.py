import asyncio
import unittest
from types import SimpleNamespace

from infra.message_broker.external_events_handlers.exam_consumer import (
    TurnEvaluationRetriesExhausted,
    _evaluate_turn_batch,
)


_NO_SEGMENTS = (
    "pronunciation_error: Azure speech recognition returned no recognized segments"
)


class ExamConsumerTurnBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_only_the_transient_failed_turn(self) -> None:
        turns = [
            SimpleNamespace(turn_order=1),
            SimpleNamespace(turn_order=2),
            SimpleNamespace(turn_order=3),
        ]
        calls = {1: 0, 2: 0, 3: 0}

        async def evaluate_one(turn):
            calls[turn.turn_order] += 1
            await asyncio.sleep(0)
            if turn.turn_order == 2 and calls[2] == 1:
                raise RuntimeError(_NO_SEGMENTS)
            return turn, {"turn_order": turn.turn_order}

        results = await _evaluate_turn_batch(
            turns,
            evaluate_one,
            transient_retry_count=2,
        )

        self.assertEqual([turn.turn_order for turn, _result in results], [1, 2, 3])
        self.assertEqual(calls, {1: 1, 2: 2, 3: 1})

    async def test_waits_for_other_turns_before_retrying_failed_turn(self) -> None:
        turns = [
            SimpleNamespace(turn_order=1),
            SimpleNamespace(turn_order=2),
        ]
        slow_turn_finished = asyncio.Event()
        retried_before_slow_turn_finished = False
        failed_turn_calls = 0

        async def evaluate_one(turn):
            nonlocal failed_turn_calls, retried_before_slow_turn_finished
            if turn.turn_order == 1:
                await asyncio.sleep(0.02)
                slow_turn_finished.set()
                return turn, {"turn_order": turn.turn_order}

            failed_turn_calls += 1
            if failed_turn_calls == 1:
                raise RuntimeError(_NO_SEGMENTS)
            retried_before_slow_turn_finished = not slow_turn_finished.is_set()
            return turn, {"turn_order": turn.turn_order}

        await _evaluate_turn_batch(
            turns,
            evaluate_one,
            transient_retry_count=1,
        )

        self.assertFalse(retried_before_slow_turn_finished)

    async def test_raises_terminal_error_after_targeted_retries(self) -> None:
        turn = SimpleNamespace(turn_order=4)
        calls = 0

        async def evaluate_one(_turn):
            nonlocal calls
            calls += 1
            raise RuntimeError(_NO_SEGMENTS)

        with self.assertRaises(TurnEvaluationRetriesExhausted):
            await _evaluate_turn_batch(
                [turn],
                evaluate_one,
                transient_retry_count=2,
            )

        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
