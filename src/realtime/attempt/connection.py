"""One WebSocket and one Voice Live client for an entire exam attempt."""

import asyncio
import logging
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

from infra import practice_session_client
from infra.alert_client import push_alert
from infra.realtime_socket import RealtimeSocket
from infra.voice_live_client import EXPECTED_SAMPLE_RATE, VoiceLiveClient, VoiceLiveServerEvent
from node.followUpDecisionGraph.constants import CLOSING_REPLY, EXAM_FAREWELL_TEXT
from realtime.question.coordinator import QuestionSessionCoordinator
from realtime.background import spawn

logger = logging.getLogger(__name__)


def _write_turn_wav(pcm16_bytes: bytes) -> Optional[str]:
    """Ghi audio thô của lượt ra WAV tạm, 16kHz/16-bit/mono.

    Bản sao có chủ ý của practice_attempt.connection._write_turn_wav: hai đường thi/luyện chạy
    độc lập và đổi một bên không được kéo theo bên kia. Cùng khuôn với WAV mà WPF từng upload,
    nên transcribe_turn_node đọc vào không thấy khác biệt gì.
    """
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
        # Chính những byte đang đẩy sang Voice Live, giữ lại một bản cho tới hết lượt.
        #
        # Trước đây hàm dưới chuyển tiếp rồi quên luôn, nên khi cần audio để phiên âm lại thì
        # Python phải TẢI VỀ từ S3 đúng file WPF vừa upload. Đường luyện tập không hề như vậy
        # (practice_attempt/connection.py:183-184). Giữ lại ở đây là đủ để bỏ hẳn vòng tải về.
        self._turn_audio_buffer = bytearray()
        # Các việc lưu lượt đang chạy nền. Client hỏi số này TRƯỚC khi nộp bài để biết còn phải
        # chờ hay không -- xem GET /realtime/attempts/{id}/pending-archives.
        #
        # Cửa chờ này trước đây nằm ở WPF (TurnArchiveQueue.DrainAsync). Từ khi audio do Python
        # lưu, hàng đợi bên WPF luôn rỗng nên chờ ở đó là chờ hư không: việc thật chạy ở đây.
        self._pending_archives: set = set()

    @property
    def pending_archive_count(self) -> int:
        return len(self._pending_archives)

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
            self._speak(self.questions.present_question(message))
        elif message_type == "turn_end":
            await self._handle_turn_end(message)
        elif message_type == "speech_budget_progress":
            await self.questions.checkpoint_speech_budget(message)
        elif message_type == "resume":
            await self._handle_resume(message)
        elif message_type == "focus_lost":
            await self._handle_focus_lost(message)
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
        # Câu mới thì audio còn sót của câu trước không còn ý nghĩa gì.
        self._turn_audio_buffer.clear()
        result = await self.questions.start_question(message)
        await self.socket.send_json(
            {"type": "question_start_ack", "answer_id": result.answer_id}
        )
        self._speak(result.section_instruction)

    async def _handle_turn_end(self, message: dict) -> None:
        # Chốt và xả buffer NGAY, trước process_turn: hàm đó gọi LLM mất vài giây, mà mic vẫn
        # chạy suốt -- không cắt ở đây thì tiếng của lượt sau lẫn vào lượt này.
        turn_audio = bytes(self._turn_audio_buffer)
        self._turn_audio_buffer.clear()

        # Chốt câu hỏi ĐANG được trả lời trước khi xử lý: complete_turn() đẩy
        # current_prompt_text sang câu kế tiếp ngay khi should_continue, nên đọc sau
        # process_turn là lưu nhầm câu hỏi cho lượt vừa xong.
        answered_session = self.questions.active_session
        answered_prompt = (
            answered_session.current_prompt_text if answered_session is not None else None
        )

        result = await self.questions.process_turn(message)
        if result is None:
            return

        session = self.questions.active_session
        if session is not None:
            # Chạy nền: lưu audio + phiên âm Azure KHÔNG được chắn câu hỏi kế tiếp.
            #
            # Hạn 120s chứ không lấy mặc định 30s của spawn: Azure STT chạy theo độ dài audio,
            # một câu trả lời 90 giây có thể mất hàng chục giây. Hết hạn là task bị huỷ, tức
            # mất bản ghi âm của lượt đó.
            task = spawn(
                self._archive_turn(
                    session,
                    result.completed_turn_order,
                    turn_audio,
                    message.get("duration_seconds"),
                    answered_prompt,
                ),
                name=f"archive_turn:{session.answer_id}:{result.completed_turn_order}",
                timeout=120.0,
            )
            self._pending_archives.add(task)
            task.add_done_callback(self._pending_archives.discard)

        decision = result.decision
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

    async def _archive_turn(
        self,
        session,
        turn_order: int,
        pcm16_bytes: bytes,
        duration_seconds,
        answered_prompt: Optional[str],
    ) -> None:
        """Lưu audio lượt vừa xong và phiên âm lại bằng Azure -- TẤT CẢ ở phía server.

        Trước 2026-08-13 việc này do WPF làm: upload S3 rồi gọi POST /turns/archive, và Python
        TẢI LẠI đúng file đó từ S3 để phiên âm. Ba vấn đề, đã đo được trên log thật:

        - phụ thuộc đường mạng của học sinh. Đây chính là lý do mô hình ấy đã bị bác khi làm
          luyện tập -- xem docstring đầu infra/practice_session_client.py.
        - vòng mạng thừa: byte đã đi qua tiến trình này trên WebSocket rồi lại tải về.
        - request đó không có retry, và có lần chạy quá 100 giây rồi bị cắt giữa chừng
          (SocketException 995), kéo theo lượt nói suýt không tới được Java.

        Không chặn luồng thi: mọi lỗi ở đây chỉ mất bản ghi âm và thẻ code-switch của Azure.
        Bản thân lượt nói đã sang Java từ pha sơ bộ của turn_publisher.publish_turn_if_new.
        """
        audio_path = await asyncio.to_thread(_write_turn_wav, pcm16_bytes)
        if audio_path is None:
            return

        try:
            audio_url = None
            try:
                upload_info = await practice_session_client.get_exam_turn_upload_url(
                    session.answer_id, turn_order
                )
                # Đọc trong thread: file WAV cỡ MB, đọc thẳng trên vòng lặp sự kiện là chặn
                # đúng lúc phiên thi đang cần phản hồi nhanh nhất.
                wav_bytes = await asyncio.to_thread(Path(audio_path).read_bytes)
                await practice_session_client.upload_turn_wav(
                    upload_info["uploadUrl"], wav_bytes
                )
                audio_url = upload_info.get("audioRef")
            except Exception:
                logger.exception(
                    "[attempt_connection] turn audio upload failed answer_id=%s turn_order=%s",
                    session.answer_id, turn_order,
                )

            # graph.invoke (sync) chứ không .ainvoke -- archive_graph compile với PostgresSaver
            # đồng bộ, aget_tuple của nó ném NotImplementedError. Đẩy sang thread giữ cho vòng
            # lặp sự kiện không bị chặn, y như archive_controller vẫn làm.
            await asyncio.to_thread(
                session.archive_graph.invoke,
                {
                    "answer_id": session.answer_id,
                    "audio_ref": audio_url,
                    "paper_item_id": session.paper_item_id,
                    "question": session.question,
                    "language": session.language,
                    "audio_path": audio_path,
                    "turn_order": turn_order,
                    "prompt_text": answered_prompt,
                    "duration_seconds": duration_seconds,
                    "status": "idle",
                },
                config={"configurable": {"thread_id": session.answer_id}},
            )
        except Exception:
            logger.exception(
                "[attempt_connection] archive turn failed answer_id=%s turn_order=%s",
                session.answer_id, turn_order,
            )
        finally:
            Path(audio_path).unlink(missing_ok=True)

    async def _handle_focus_lost(self, message: dict) -> None:
        """Thí sinh rời khỏi cửa sổ thi (WPF WindowFocusGuard bắt Window.Deactivated).

        Đi nhờ WS sẵn có thay vì mở endpoint REST mới: kết nối này vốn đã mở suốt buổi thi và
        đã xác thực theo exam_attempt_id, nên không phải thêm đường vào nào cũng không phải
        phát token thứ hai. Java hoàn toàn không có client AlertService -- Python là nơi DUY
        NHẤT trong hệ đã nối được tới AlertService của vox-streaming.

        `capturedAt` do client gửi lên chỉ để ghi log: đồng hồ máy học sinh không đáng tin và
        có thể bị chỉnh, nên mốc thời gian đi vào cảnh báo là mốc do alert_client tự đóng dấu
        phía server.

        KHÔNG đi qua should_emit_alert/cooldown của proctoring_alert_policy: cái đó gộp theo
        điều kiện KÉO DÀI (khuôn mặt khuất suốt 10 giây), còn đây là sự kiện RỜI RẠC -- rời đi
        ba lần là ba lần đáng ghi, gộp lại thì mất đúng thứ có giá trị kỷ luật. Việc chống spam
        đã làm ở đầu WPF (gộp trong 3 giây).
        """
        logger.warning(
            "[attempt_connection] focus_lost exam_attempt_id=%s captured_at=%s",
            self.exam_attempt_id,
            message.get("capturedAt"),
        )
        # spawn thay vì await: đẩy cảnh báo là việc phụ, không được để một lời gọi gRPC chậm
        # chặn vòng lặp đọc WS đang phục vụ âm thanh của bài thi.
        spawn(
            push_alert(
                room_id=self.exam_attempt_id,
                participant_id=self.exam_attempt_id,
                stream_id=self.exam_attempt_id,
                alert_type="WINDOW_FOCUS_LOST",
                confidence=1.0,
            )
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
        spawn(
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
