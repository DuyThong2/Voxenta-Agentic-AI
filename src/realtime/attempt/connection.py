"""One WebSocket and one Voice Live client for an entire exam attempt."""

import asyncio
import logging
from typing import Any, Optional

from infra.realtime_socket import RealtimeSocket
from infra.voice_live_client import VoiceLiveClient, VoiceLiveServerEvent
from node.followUpDecisionGraph.constants import CLOSING_REPLY, EXAM_FAREWELL_TEXT
from realtime.question.coordinator import QuestionSessionCoordinator

logger = logging.getLogger(__name__)


class AttemptConnection:
    def __init__(
        self,
        *,
        exam_attempt_id: str,
        socket: RealtimeSocket,
        archive_graph: Any,
        text_followup_graph: Any,
    ) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.socket = socket
        self.questions = QuestionSessionCoordinator(
            exam_attempt_id=exam_attempt_id,
            archive_graph=archive_graph,
            text_followup_graph=text_followup_graph,
        )
        self.voice_live_client = VoiceLiveClient(on_event=self._on_voice_live_event)
        self._utterance_sequence = 0

    async def start(self) -> None:
        await self.voice_live_client.start()

    async def handle_audio_frame(self, data: bytes) -> None:
        await self.voice_live_client.push_audio(data)

    async def handle_message(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "question_start":
            await self._handle_question_start(message)
        elif message_type == "present_question":
            self._speak(self.questions.present_question(message))
        elif message_type == "turn_end":
            await self._handle_turn_end(message)
        elif message_type == "speech_budget_progress":
            await self.questions.checkpoint_speech_budget(message)
        elif message_type == "resume":
            await self._handle_resume(message)
        elif message_type == "exam_end":
            self._speak(EXAM_FAREWELL_TEXT)
            await self.socket.send_json({"type": "exam_end_ack"})
        else:
            logger.warning(
                "[attempt_connection] unknown message type=%s exam_attempt_id=%s",
                message_type,
                self.exam_attempt_id,
            )

    async def _handle_question_start(self, message: dict) -> None:
        result = await self.questions.start_question(message)
        await self.socket.send_json(
            {"type": "question_start_ack", "answer_id": result.answer_id}
        )
        self._speak(result.section_instruction)

    async def _handle_turn_end(self, message: dict) -> None:
        result = await self.questions.process_turn(message)
        if result is None:
            return

        decision = result.decision
        session = self.questions.active_session
        await self.socket.send_json(
            {
                "type": "decision",
                "answer_id": session.answer_id if session else None,
                "decision": decision,
            }
        )
        next_prompt = decision.get("next_prompt_text") or (
            None if decision.get("should_continue") else CLOSING_REPLY
        )
        self._speak(
            next_prompt,
            slow=decision.get("reason") in {"clarify_prompt", "decline_repair"},
        )

    async def _handle_resume(self, message: dict) -> None:
        result = await self.questions.resume(message)
        await self.socket.send_json(result.acknowledgement)

        if result.recovered_decision is not None:
            decision = result.recovered_decision
            next_prompt = decision.get("next_prompt_text") or (
                None if decision.get("should_continue") else CLOSING_REPLY
            )
            self._speak(next_prompt)
        else:
            self._speak(result.prompt_to_speak)

    def _speak(self, text: Optional[str], *, slow: bool = False) -> None:
        self._utterance_sequence += 1
        asyncio.create_task(
            self._send_speak(text or "", self._utterance_sequence, slow)
        )

    async def _send_speak(self, text: str, sequence: int, slow: bool) -> None:
        session = self.questions.active_session
        logger.info(
            "[realtime_ai_speech] exam_attempt_id=%s answer_id=%s turn_order=%s sequence=%d text=%r",
            self.exam_attempt_id,
            session.answer_id if session else None,
            session.turn_order if session else None,
            sequence,
            text,
        )
        await self.socket.send_json(
            {
                "type": "speak",
                "sequence": sequence,
                "text": text,
                "rate": "-20%" if slow else None,
            }
        )

    async def _on_voice_live_event(self, event: VoiceLiveServerEvent) -> None:
        routed = self.questions.route_voice_event(event)
        if not routed and event.kind in {
            "vad_speech_start",
            "vad_speech_end",
            "partial_transcript",
            "final_transcript",
        }:
            logger.warning(
                "[attempt_connection] event=%s without active question exam_attempt_id=%s",
                event.kind,
                self.exam_attempt_id,
            )

        payload = {"type": event.kind}
        if event.text is not None:
            payload["text"] = event.text
        await self.socket.send_json(payload)

    async def force_end(self, reason: Optional[str]) -> None:
        await self.socket.send_json(
            {"type": "force_end", "reason": reason or ""}
        )
        await self.socket.close(code=1000)

    async def close(self) -> None:
        self.questions.clear()
        await self.voice_live_client.close()
