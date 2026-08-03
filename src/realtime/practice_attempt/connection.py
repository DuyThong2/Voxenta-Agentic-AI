"""One WebSocket and one Voice Live client for an entire practice session -- parallel to
realtime/attempt/connection.py (gói 11 mục 2.3).

Biggest behavioral difference from exam, per the confirmed click-to-continue UX (mục 2.2):
this class never auto-speaks the next prompt after a turn ends, for EITHER a follow-up or a
brand new MAIN question -- the client always sends `present_question` itself, only after the
student taps "Tiếp tục". Exam's AttemptConnection speaks next_prompt_text immediately on
should_continue=True and a closing line on should_continue=False; neither happens here.

The other new responsibility: after every turn (follow-up or not), this concurrently (a) runs
realtimeCorrectionGraph for feedback, (b) submits the turn to Java (records it + consumes
quota + triggers grading on the last turn of a question), and (c) if the follow-up chain just
ended, asks Java for the next MAIN question -- pushing `correction` and `next_question` as
independent messages the moment each is ready (see mục 2.4c/2.6, "đẩy độc lập").
"""

import asyncio
import json
import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from infra import practice_session_client
from infra.realtime_socket import RealtimeSocket
from infra.voice_live_client import EXPECTED_SAMPLE_RATE, VoiceLiveClient, VoiceLiveServerEvent
from node.realtimeCorrectionGraph.graphConfig import build_realtime_correction_graph
from realtime.practice_attempt.coordinator import PracticeQuestionSessionCoordinator

logger = logging.getLogger(__name__)

PRACTICE_FAREWELL_TEXT = "Cảm ơn bạn đã luyện tập hôm nay. Hẹn gặp lại!"

_correction_graph = build_realtime_correction_graph()


@dataclass(frozen=True)
class TurnSubmitOutcome:
    """Ket qua nop mot luot len Java: trang thai + cap so cho thanh tien do.

    spoken/budget la None khi khong doc duoc tu Java (loi mang, hoac ban Java cu chua co 2
    truong nay). Co y KHONG thay bang 0: 0 la mot con so co nghia ("chua noi giay nao"), con
    None la "khong biet" -- client an thanh tien do thay vi ve mot thanh rong sai su that.
    """

    status: str
    spoken_seconds: Optional[int]
    budget_seconds: Optional[int]

    @classmethod
    def of(cls, status: str, body: Any) -> "TurnSubmitOutcome":
        if not isinstance(body, dict):
            return cls(status, None, None)
        return cls(
            status,
            body.get("sessionSpokenSeconds"),
            body.get("sessionBudgetSeconds"),
        )


def _write_turn_wav(pcm16_bytes: bytes) -> Optional[str]:
    """Writes this turn's raw mic audio (already buffered from handle_audio_frame -- the
    same bytes pushed to Voice Live, not a second upload) to a temp WAV file so
    PronunciationNode can run Azure Pronunciation Assessment against it. Matches
    voice_live_client.EXPECTED_SAMPLE_RATE (16kHz/16-bit/mono, same format
    node/followUpDecisionGraph/graphConfig.py._wav_duration_seconds assumes for exam's
    WPF-uploaded WAVs) -- caller is responsible for deleting the file once done with it."""
    if not pcm16_bytes:
        return None
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.close()
    with wave.open(tmp_file.name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(EXPECTED_SAMPLE_RATE)
        wav_file.writeframes(pcm16_bytes)
    return tmp_file.name


class PracticeAttemptConnection:
    def __init__(
        self,
        *,
        practice_session_id: str,
        socket: RealtimeSocket,
        archive_graph: Any,
        text_followup_graph: Any,
    ) -> None:
        self.practice_session_id = practice_session_id
        self.socket = socket
        self.questions = PracticeQuestionSessionCoordinator(
            practice_session_id=practice_session_id,
            archive_graph=archive_graph,
            text_followup_graph=text_followup_graph,
        )
        self.voice_live_client = VoiceLiveClient(on_event=self._on_voice_live_event)
        self._utterance_sequence = 0
        # Raw PCM16 for the turn currently in flight -- same bytes pushed to Voice Live below,
        # not a second capture path. Reset in _handle_turn_end once handed off to
        # _run_correction, so it never spans more than one turn.
        self._turn_audio_buffer = bytearray()

    async def start(self) -> None:
        await self.voice_live_client.start()

    async def handle_audio_frame(self, data: bytes) -> None:
        self._turn_audio_buffer.extend(data)
        await self.voice_live_client.push_audio(data)

    async def handle_message(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "question_start":
            await self._handle_question_start(message)
        elif message_type == "present_question":
            # Discard whatever accumulated in the mic buffer while the student was reading
            # feedback and deciding to tap "Tiếp tục" (audio keeps streaming continuously,
            # see practice_session_screen.dart) -- otherwise that dead air/ambient noise
            # would prepend onto the NEXT turn's WAV once turn_end fires.
            self._turn_audio_buffer.clear()
            self._speak(self.questions.present_question(message))
        elif message_type == "ready_to_answer":
            # Client-local signal that TTS actually finished speaking the prompt (see
            # _onAiSpeechDone in practice_session_screen.dart) -- Python has no other way to
            # know that moment, so the buffer clear at present_question above still leaves the
            # whole TTS-playback duration (silence/ambient noise, possibly echo on
            # speaker+mic devices) sitting at the front of the WAV. Re-clearing here shrinks
            # that window down to just the (short, unavoidable) barge-in reaction latency.
            self._turn_audio_buffer.clear()
        elif message_type == "turn_end":
            await self._handle_turn_end(message)
        elif message_type == "speech_budget_progress":
            await self.questions.checkpoint_speech_budget(message)
        elif message_type == "resume":
            await self._handle_resume(message)
        elif message_type == "practice_end":
            self._speak(PRACTICE_FAREWELL_TEXT)
            await self.socket.send_json({"type": "practice_end_ack"})
        else:
            logger.warning(
                "[practice_attempt_connection] unknown message type=%s practice_session_id=%s",
                message_type,
                self.practice_session_id,
            )

    async def _handle_question_start(self, message: dict) -> None:
        result = await self.questions.start_question(message)
        await self.socket.send_json(
            {"type": "question_start_ack", "answer_id": result.answer_id}
        )
        # No section_instruction speech here on purpose: unlike exam (auto-continue), the
        # first MAIN question's prompt is only spoken once the client sends
        # present_question -- same click-to-continue rule as every later transition.

    async def _handle_turn_end(self, message: dict) -> None:
        result = await self.questions.process_turn(message)
        if result is None:
            return

        # Snapshot + reset right away so the NEXT turn's audio (which may already be
        # streaming in while this turn's correction/submit is still running in the
        # background task below) never bleeds into this turn's WAV.
        turn_audio = bytes(self._turn_audio_buffer)
        self._turn_audio_buffer.clear()

        decision = result.decision
        session = self.questions.active_session
        await self.socket.send_json(
            {
                "type": "decision",
                "answer_id": session.answer_id if session else None,
                "decision": decision,
            }
        )
        # Deliberately no self._speak(...) here -- see module docstring. The next prompt
        # (follow-up or new MAIN question) is only spoken when the client sends
        # present_question after the student taps "Tiếp tục".
        asyncio.create_task(self._after_turn(session, decision, turn_audio))

    async def _after_turn(self, session, decision: dict, turn_audio: bytes) -> None:
        if session is None or not session.turns:
            return
        should_continue = bool(decision.get("should_continue"))
        current_turn = session.turns[-1]

        # Written once, shared read-only by both the correction graph (Pronunciation
        # Assessment) and the S3 upload below -- deleted only after both are done with it.
        audio_path = await asyncio.to_thread(_write_turn_wav, turn_audio)
        try:
            correction_task = asyncio.create_task(self._run_correction(session, audio_path))
            upload_task = asyncio.create_task(
                self._upload_turn_audio(audio_path, current_turn.get("turn_order"))
            )

            corrections, pronunciation_result = await correction_task
            audio_url = await upload_task
            # Only resolve/push the next MAIN question -- which mutates the paper on Java's
            # side -- once the CURRENT turn is confirmed saved (recorded + quota-consumed).
            # Firing it concurrently with _submit_turn let the session silently advance past
            # a turn that was never recorded (network hiccup or quota exceeded).
            submit_outcome = await self._submit_turn(
                session, corrections, pronunciation_result, audio_url,
                question_complete=not should_continue,
            )
            await self.socket.send_json(
                {
                    "type": "correction",
                    "answer_id": session.answer_id,
                    "corrections": corrections,
                    "pronunciation": pronunciation_result,
                }
            )
            await self._send_session_budget(
                submit_outcome.spoken_seconds, submit_outcome.budget_seconds
            )

            if submit_outcome.status in ("quota_exceeded", "failed"):
                # Both are unrecoverable mid-conversation: Java's paper/quota state and the
                # client's UI have now diverged (this turn is unsaved or rejected), so letting
                # the student tap "Tiếp tục" as if nothing happened would silently continue past
                # data loss. End the session cleanly instead -- same shape as budget_exhausted/
                # pool_exhausted (mục 2.9), just a different reason for the summary screen to
                # explain.
                await self.socket.send_json(
                    {"type": "practice_session_ended", "reason": submit_outcome.status}
                )
                return

            if not should_continue:
                # Cau CUOI da duoc Java may do vua dung ngan sach con lai -- tra loi xong
                # la het gio, hoi tiep chi de nhan lai budget_exhausted, ton mot vong goi
                # va bat hoc sinh cho them vai giay truoc man tong ket.
                if getattr(session, "last_question", False):
                    await self.socket.send_json(
                        {"type": "practice_session_ended", "reason": "budget_exhausted"}
                    )
                    return
                await self._resolve_and_push_next_question()
        finally:
            if audio_path:
                Path(audio_path).unlink(missing_ok=True)

    async def _run_correction(self, session, audio_path: Optional[str]) -> tuple[list, Optional[dict]]:
        current_turn = session.turns[-1]
        state = {
            "transcript": current_turn.get("transcript") or "",
            "audio_path": audio_path,
            "language": session.language,
        }
        try:
            result = await asyncio.to_thread(_correction_graph.invoke, state)
        except Exception:
            logger.exception(
                "[practice_attempt_connection] correction graph failed practice_session_id=%s",
                self.practice_session_id,
            )
            return [], None
        return result.get("corrections") or [], result.get("pronunciation_result")

    async def _upload_turn_audio(self, audio_path: Optional[str], turn_order: int) -> Optional[str]:
        """Permanent archival for teacher review (TeacherPracticeTurnView.audioUrl) -- separate
        purpose from the temp WAV's other use (Pronunciation Assessment input, deleted after).
        Mirrors WPF's own upload-url-then-PUT pattern (GetTurnUploadUrlUseCase), just server-side:
        Java never sees the audio bytes, only mints the presigned S3 URL."""
        if not audio_path:
            return None
        try:
            upload_info = await practice_session_client.get_turn_upload_url(
                self.practice_session_id, turn_order
            )
            wav_bytes = await asyncio.to_thread(Path(audio_path).read_bytes)
            await practice_session_client.upload_turn_wav(upload_info["uploadUrl"], wav_bytes)
            return upload_info.get("audioRef")
        except httpx.HTTPError:
            logger.exception(
                "[practice_attempt_connection] turn audio upload failed practice_session_id=%s turn_order=%s",
                self.practice_session_id,
                turn_order,
            )
            return None

    async def _submit_turn(
        self,
        session,
        corrections: list,
        pronunciation_result: Optional[dict],
        audio_url: Optional[str],
        *,
        question_complete: bool,
    ) -> TurnSubmitOutcome:
        """status la "ok", "quota_exceeded" (Java bao het han muc) hoac "failed" -- caller
        (_after_turn) dua vao do de quyet co an toan di tiep khong.

        Kem theo cap so "da noi / ngan sach" Java tra ve trong body 200, de client cap nhat
        thanh tien do ngay sau moi luot. None khi khong nop duoc (khong bia so)."""
        current_turn = session.turns[-1]
        payload = {
            "questionId": session.question_id,
            "turnOrder": current_turn.get("turn_order"),
            "turnType": current_turn.get("turn_type"),
            "promptText": current_turn.get("prompt_text"),
            "audioUrl": audio_url,
            "transcript": current_turn.get("transcript") or "",
            "durationSeconds": int(current_turn.get("duration_seconds") or 0),
            "wordFeedbackJson": json.dumps(pronunciation_result) if pronunciation_result else None,
            "turnScore": None,
            "questionComplete": question_complete,
            "corrections": [
                {
                    "category": c.get("category", "grammar"),
                    "originalText": c.get("original_text", ""),
                    "correctedText": c.get("corrected_text", ""),
                    "explanation": c.get("explanation", ""),
                    "correctAudioUrl": c.get("correct_audio_url"),
                    "confidence": c.get("confidence", 0.0),
                }
                for c in corrections
            ],
        }
        # A turn that fails to save is unrecoverable client-side (the recorded audio/corrections
        # already got handed off) -- retry transient failures here, server-to-server, rather than
        # push the problem to the student. 409 (quota exceeded) is a definitive business
        # rejection, not transient, so it's never retried.
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                result = await practice_session_client.submit_turn(
                    self.practice_session_id, payload
                )
                # Hết hạn mức giờ báo bằng CỜ trong body 200, không phải 409 nữa: Java vẫn
                # GHI lượt rồi mới báo hết quota, thay vì rollback cả lượt vừa nói. Nhánh 409
                # bên dưới giữ lại để tương thích khi Java chưa kịp deploy bản mới.
                if isinstance(result, dict) and result.get("quotaExhausted"):
                    logger.info(
                        "[practice_attempt_connection] quota exhausted -- luot da duoc ghi, "
                        "dong phien practice_session_id=%s answer_id=%s",
                        self.practice_session_id,
                        session.answer_id,
                    )
                    return TurnSubmitOutcome.of("quota_exceeded", result)
                return TurnSubmitOutcome.of("ok", result)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 409:
                    logger.warning(
                        "[practice_attempt_connection] quota exceeded (409, ban Java cu -- "
                        "luot NAY BI MAT) practice_session_id=%s answer_id=%s",
                        self.practice_session_id,
                        session.answer_id,
                    )
                    return TurnSubmitOutcome("quota_exceeded", None, None)
                logger.exception(
                    "[practice_attempt_connection] submit_turn failed (attempt %d/%d) "
                    "practice_session_id=%s answer_id=%s",
                    attempt,
                    attempts,
                    self.practice_session_id,
                    session.answer_id,
                )
            except httpx.HTTPError:
                logger.exception(
                    "[practice_attempt_connection] submit_turn failed (attempt %d/%d) "
                    "practice_session_id=%s answer_id=%s",
                    attempt,
                    attempts,
                    self.practice_session_id,
                    session.answer_id,
                )
            if attempt < attempts:
                await asyncio.sleep(1.5 * attempt)
        return TurnSubmitOutcome("failed", None, None)

    async def _send_session_budget(
        self, spoken_seconds: Optional[int], budget_seconds: Optional[int]
    ) -> None:
        """Cap nhat thanh tien do "da noi / ngan sach" tren may hoc sinh.

        Im lang bo qua khi thieu so: xem TurnSubmitOutcome -- None nghia la khong biet,
        gui xuong se lam client ve mot thanh sai su that. Client giu nguyen so cu.

        Luu y day KHONG phai dong ho: quota chi tru dung khoang VAD nghe thay tieng, nen
        luc AI noi / hoc sinh nghi / cho cham deu khong tinh vao day.
        """
        if spoken_seconds is None or budget_seconds is None:
            return
        await self.socket.send_json(
            {
                "type": "session_budget",
                "spoken_seconds": spoken_seconds,
                "budget_seconds": budget_seconds,
            }
        )

    async def _resolve_and_push_next_question(self) -> None:
        push_result = await self.questions.resolve_and_push_next_question()
        await self._send_session_budget(
            push_result.session_spoken_seconds, push_result.session_budget_seconds
        )
        if push_result.ended:
            await self.socket.send_json(
                {"type": "practice_session_ended", "reason": push_result.reason}
            )
            return
        await self.socket.send_json(
            {
                "type": "next_question",
                "answer_id": push_result.answer_id,
                "question": push_result.question,
            }
        )

    async def _handle_resume(self, message: dict) -> None:
        result = await self.questions.resume(message)
        await self.socket.send_json(result.acknowledgement)
        # No auto-speak on resume either (click-to-continue) -- the ack carries
        # prompt_to_speak (if any) for the client to re-send present_question itself, same as
        # every other transition.
        if result.recovered_decision is not None:
            # A turn_end WAS archived before the connection dropped, but this connection
            # object (and its in-memory _turn_audio_buffer) died with it before _after_turn
            # ever ran -- correction/submit/resolve-next-question never happened for that
            # turn. The raw audio is unrecoverable (it lived only in the dead connection's
            # buffer), so re-run _after_turn with empty audio: _write_turn_wav/
            # _run_correction/_upload_turn_audio all already no-op cleanly on no audio (see
            # their own guards), but the turn still gets recorded/quota-consumed/graded from
            # its transcript, and the next MAIN question still gets resolved if needed --
            # rather than silently stalling the session forever.
            asyncio.create_task(
                self._after_turn(self.questions.active_session, result.recovered_decision, b"")
            )

    def _speak(self, text: Optional[str], *, slow: bool = False) -> None:
        self._utterance_sequence += 1
        asyncio.create_task(
            self._send_speak(text or "", self._utterance_sequence, slow)
        )

    async def _send_speak(self, text: str, sequence: int, slow: bool) -> None:
        session = self.questions.active_session
        logger.info(
            "[practice_realtime_ai_speech] practice_session_id=%s answer_id=%s turn_order=%s sequence=%d text=%r",
            self.practice_session_id,
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
                "[practice_attempt_connection] event=%s without active question practice_session_id=%s",
                event.kind,
                self.practice_session_id,
            )

        payload = {"type": event.kind}
        if event.text is not None:
            payload["text"] = event.text
        await self.socket.send_json(payload)

    async def force_end(self, reason: Optional[str]) -> None:
        await self.socket.send_json({"type": "force_end", "reason": reason or ""})
        await self.socket.close(code=1000)

    async def close(self) -> None:
        self.questions.clear()
        await self.voice_live_client.close()
