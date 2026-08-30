"""Reconnect bookkeeping for the YOLO proctoring WebRTC session.

The scenario under test is a student's network dropping mid-exam and the desktop client
reconnecting. Python keys the proctoring session on ``exam_attempt_id``, so the reconnect lands on
the SAME ``session_id`` as the connection it is replacing -- which is what makes the ordering here
subtle enough to be worth pinning down.
"""

import unittest

from infra.webrtc import proctoring_session


class FakePeerConnection:
    """Stands in for RTCPeerConnection: eviction only closes it and compares identity."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class ProctoringSessionReconnectTests(unittest.IsolatedAsyncioTestCase):
    SESSION_ID = "exam-attempt-1"

    def setUp(self) -> None:
        self._reset()

    def tearDown(self) -> None:
        self._reset()

    @staticmethod
    def _reset() -> None:
        # Module-level registries, so a leaked entry would surface as a failure in an unrelated test.
        proctoring_session.pcs.clear()
        proctoring_session.session_map.clear()
        proctoring_session.session_events.clear()
        proctoring_session.session_subscribers.clear()
        proctoring_session.session_identity.clear()

    def _register(self, pc: FakePeerConnection) -> None:
        proctoring_session.pcs.add(pc)
        proctoring_session.session_map[self.SESSION_ID] = pc

    async def test_evicting_closes_the_previous_peer_and_frees_the_key(self) -> None:
        previous = FakePeerConnection("previous")
        self._register(previous)

        evicted = await proctoring_session.evict_previous_connection(self.SESSION_ID)

        self.assertTrue(evicted)
        self.assertTrue(previous.closed)
        self.assertNotIn(previous, proctoring_session.pcs)
        self.assertNotIn(self.SESSION_ID, proctoring_session.session_map)

    async def test_evicting_keeps_the_event_log_and_identity(self) -> None:
        """A reconnect mid-exam is not the end of the exam session.

        This is the whole reason eviction is separate from ``cleanup_session``: running the full
        teardown would push SESSION_ENDED to live SSE subscribers, drop the event log, and reset the
        alert policy's streak/cooldown state -- so every network blip would re-arm duplicate alerts
        the student has already been warned about.
        """
        proctoring_session.session_events[self.SESSION_ID].append(
            proctoring_session.build_event("PHONE_DETECTED", "phone in frame")
        )
        proctoring_session.register_identity(
            self.SESSION_ID, exam_session_id="session-1", participant_id="candidate-1"
        )
        self._register(FakePeerConnection("previous"))

        await proctoring_session.evict_previous_connection(self.SESSION_ID)

        self.assertEqual(len(proctoring_session.session_events[self.SESSION_ID]), 1)
        self.assertEqual(
            proctoring_session.get_identity(self.SESSION_ID)["participant_id"], "candidate-1"
        )

    async def test_superseded_peer_cannot_tear_down_its_replacement(self) -> None:
        """The regression this guard exists for.

        The exam machine notices the outage first -- it is actively watching -- so it reconnects
        within a second or two, while aiortc on this side only finds out when ICE consent expires.
        The old peer's ``connectionstatechange`` handler is therefore still to come, and it closes
        over the same ``session_id`` the new connection now owns. Without the identity check it
        would run ``cleanup_session`` on the replacement, and the two sides would loop: reconnect,
        die a few seconds later, reconnect.
        """
        previous = FakePeerConnection("previous")
        self._register(previous)

        await proctoring_session.evict_previous_connection(self.SESSION_ID)
        current = FakePeerConnection("current")
        self._register(current)

        # What the handler in controller/webrtc.py asks before tearing anything down.
        self.assertFalse(proctoring_session.is_current_connection(self.SESSION_ID, previous))
        self.assertTrue(proctoring_session.is_current_connection(self.SESSION_ID, current))
        self.assertFalse(current.closed)

    async def test_evicting_an_unknown_session_is_a_no_op(self) -> None:
        """The ordinary first connection of an exam takes this path."""
        self.assertFalse(await proctoring_session.evict_previous_connection("never-seen"))

    async def test_a_peer_that_was_never_superseded_is_still_current(self) -> None:
        """The guard must not block the case it is not there for: an ordinary disconnect with no
        reconnect behind it still has to clean itself up."""
        only = FakePeerConnection("only")
        self._register(only)

        self.assertTrue(proctoring_session.is_current_connection(self.SESSION_ID, only))


if __name__ == "__main__":
    unittest.main()
