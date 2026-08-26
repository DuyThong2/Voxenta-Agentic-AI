"""One WebSocket and one Voice Live client for an entire exam attempt."""

import asyncio
import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from infra import practice_session_client
from infra.alert_client import push_alert
from infra.message_broker import ai_usage_tracker
from infra.realtime_socket import RealtimeSocket
from infra.voice_live_client import EXPECTED_SAMPLE_RATE, VoiceLiveClient, VoiceLiveServerEvent
from node.followUpDecisionGraph.constants import CLOSING_REPLY, EXAM_FAREWELL_TEXT
from realtime.question.coordinator import QuestionSessionCoordinator
from realtime.background import spawn

logger = logging.getLogger(__name__)

# Nhip tim 5 giay: client coi la mat ket noi khi qua 3 nhip (15 giay) khong nghe thay gi. Dat thua
# du so voi tan suat nay de mot goi ping tre khong lam dong ho dung oan.
HEARTBEAT_INTERVAL_SECONDS = 5


@dataclass(frozen=True)
class DeliveryState:
    """Thứ cần mang sang khi một kết nối bị thay bằng kết nối mới của cùng lượt thi.

    CHỈ sống trong tiến trình. Nếu client nối lại trúng pod khác (agents chạy nhiều replica, và
    sticky session của ALB dựa vào cookie nên client không phải trình duyệt sẽ trượt) thì trạng
    thái này mất và câu hỏi vẫn có thể bị phát lại một lần. Bản vá bền vững cần client gửi kèm
    `last_heard_turn_order` trong bản tin resume -- chỉ client mới biết cái gì thật sự phát ra loa.
    """

    sequence: int
    last_delivered_prompt: Optional[str]


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
        delivery_state: Optional["DeliveryState"] = None,
    ) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.socket = socket
        self.questions = QuestionSessionCoordinator(
            exam_attempt_id=exam_attempt_id,
            archive_graph=archive_graph,
            text_followup_graph=text_followup_graph,
        )
        self.voice_live_client = VoiceLiveClient(on_event=self._on_voice_live_event)
        # Nối tiếp bộ đếm của kết nối vừa bị đuổi, KHÔNG bắt đầu lại từ 0.
        #
        # `sequence` là thứ duy nhất client có để loại câu nói trùng. Đếm lại từ 1 ở mỗi kết nối
        # nghĩa là cùng một câu tới client dưới dạng `sequence=1` ở kết nối A và `sequence=1` ở
        # kết nối B -- client không có cách nào phân biệt bản trùng với câu mới. Đo thật
        # 2026-08-26: đúng tình huống đó, hai câu hỏi giống hệt cùng mang sequence=1.
        self._utterance_sequence = delivery_state.sequence if delivery_state else 0
        # Câu đã GỬI ĐI THÀNH CÔNG gần nhất, để lần resume sau không phát lại y nguyên.
        #
        # Ghi sau khi `send_json` trả về, KHÔNG ghi lúc sắp gửi: ca hỏng thật là `send_json` ném
        # ClientDisconnected -- câu đó chưa từng tới tai thí sinh nên PHẢI được phát lại.
        self._last_delivered_prompt: Optional[str] = (
            delivery_state.last_delivered_prompt if delivery_state else None
        )
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
        # Các lượt đã lưu trữ xong nhưng KHÔNG có bản ghi âm (bộ đệm rỗng lúc turn_end).
        #
        # Có mặt ở đây vì client cần phân biệt "chưa lưu xong" với "vĩnh viễn không có audio", mà
        # trạng thái bền không trả lời được: đường ghi vào `turns` nằm SAU chỗ thoát sớm trong
        # _archive_turn, nên lượt kiểu này không bao giờ xuất hiện ở đó. Không có dấu này thì
        # client hỏi tới hết hạn rồi mới chịu nộp bài -- xem get_pending_archives.
        #
        # KHÔNG ghi vào `turns` để đánh dấu: pha 2 của turn_publisher.publish_turn_if_new đang
        # chờ đúng danh sách đó rồi publish ĐÈ lên bản sơ bộ, nên một lượt tổng hợp không audio
        # sẽ ghi đè transcript realtime đang tốt bằng chuỗi rỗng (ca đã đo: 15 giây, 37 từ,
        # audio_url null). Mất dữ liệu thật, đổi lấy việc đỡ chờ -- không đáng.
        #
        # Khoá theo (answer_id, turn_order) chứ không riêng turn_order: một kết nối chạy qua
        # nhiều câu hỏi, mà turn_order đếm lại từ 1 ở mỗi câu.
        self._turns_without_audio: set[tuple[str, int]] = set()
        # Nhip tim gui xuong client. Xem _heartbeat_loop.
        #
        # GIU THAM CHIEU vao self: event loop chi giu tham chieu YEU toi task, nen mot task chi
        # duoc `create_task(...)` roi vut di co the bi thu gom giua chung -- khong loi, khong log.
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def pending_archive_count(self) -> int:
        return len(self._pending_archives)

    def is_resolved_without_audio(self, answer_id: str, turn_order: int) -> bool:
        """Lượt này đã lưu trữ xong và chắc chắn KHÔNG có bản ghi âm hay chưa.

        Chỉ đúng trong tiến trình này; pod restart là mất. Chấp nhận được vì cùng lúc đó
        get_attempt_connection cũng trả None và endpoint bỏ hẳn phần trả lời -- client rơi về
        cách đếm cũ chứ không kẹt.
        """
        return (answer_id, turn_order) in self._turns_without_audio

    async def start(self) -> None:
        await self.voice_live_client.start()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Bao cho client biet ket noi con song, deu dan va vo dieu kien.

        Vi sao can: giao thuc nay im lang mot cach HOP LE trong luc thi sinh dang nghi -- server
        chi noi khi co viec (ack, su kien VAD, quyet dinh follow-up). Nen phia client khong the
        lay "khong nhan duoc gi" lam dau hieu mat mang, va no cung khong the tin vao trang thai
        socket: rut day mang thi TCP van giu socket "mo" cho toi khi het thoi gian truyen lai, co
        the hang phut.

        Co nhip tim thi client phan biet duoc hai kieu im lang, va dung dong ho thi khi mat ket
        noi thay vi tru oan thoi gian cua thi sinh (ExamViewModel doc IsRealtimeAlive).
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                await self.socket.send_json({"type": "ping"})
        except asyncio.CancelledError:
            raise
        except Exception:
            # Socket dut la chuyen binh thuong o day; vong lap ket thuc, khong can bao dong.
            logger.debug(
                "[attempt_connection] heartbeat dung exam_attempt_id=%s", self.exam_attempt_id
            )

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
        elif message_type == "camera_signal_lost":
            await self._handle_camera_signal_lost(message)
        elif message_type == "camera_signal_restored":
            await self._handle_camera_signal_restored(message)
        elif message_type == "asset_playback_failed":
            self._handle_asset_playback_failed(message)
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
        #
        # Clear vô điều kiện là CÓ CHỦ ĐÍCH. question_start chỉ được gửi từ hai chỗ ở client:
        # PresentInitialAsync (câu mới -- đệm đang chứa tiếng thừa của câu trước, clear là đúng) và
        # PresentResumeAsync (vào lại giữa câu -- kết nối vừa lập nên đệm đã rỗng, clear là no-op).
        # Để nó phá audio thật thì message này phải tới SAU khi thí sinh bắt đầu trả lời, mà thứ tự
        # phía client chặn: question_start -> AI đọc đề -> mới mở cửa sổ mic.
        #
        # Làm nó có điều kiện ("chỉ clear khi answer_id đổi") là đổi rủi ro lấy rủi ro: tiếng còn
        # sót từ trước sẽ lẫn vào lượt mới. Nên thay vì đoán, ĐO: log lại mỗi lần clear vứt đi một
        # bộ đệm không rỗng. Thấy dòng này kèm số byte lớn nghĩa là lập luận trên có lỗ -- lúc đó
        # mới có căn cứ để đổi.
        discarded_bytes = len(self._turn_audio_buffer)
        if discarded_bytes > 0:
            logger.info(
                "[attempt_connection] question_start xoa bo dem audio con %d byte "
                "exam_attempt_id=%s answer_id=%s",
                discarded_bytes,
                self.exam_attempt_id,
                message.get("answer_id"),
            )
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
        if answered_session is not None:
            duration_ms = self.voice_live_client.pop_input_audio_duration_ms()
            if duration_ms > 0:
                ai_usage_tracker.record_duration_usage(
                    answered_session.answer_id, "azure_voice_live_input", duration_ms
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
            # _write_turn_wav chỉ trả None khi pcm16_bytes RỖNG, tức bộ đệm audio của lượt trống
            # đúng lúc turn_end tới. Lượt vẫn có transcript (Voice Live ghi thẳng xuống Postgres
            # theo answer_id), nên nhìn từ ngoài thì bài "có nói mà không có bản ghi âm".
            #
            # Trước bản này chỗ này return im lặng, không một dòng log -- nên khi chuyện đó xảy ra
            # thật (2026-08-18, answer c65800d0 turn 1: 15 giây, 37 từ, audio_url null) không có
            # cách nào truy vì sao. Ghi lại đủ số để lần sau đối chiếu được.
            #
            # Bộ đệm là trạng thái trong RAM của MỘT đối tượng AttemptConnection, còn transcript
            # thì bền trong DB. Nên mọi việc thay/đặt lại kết nối (client reconnect giữa câu) hoặc
            # một message question_start tới muộn (_handle_question_start clear buffer) đều cho ra
            # đúng dấu hiệu này.
            logger.warning(
                "[attempt_connection] bo dem audio RONG luc turn_end -- mat ban ghi am, "
                "van con transcript. answer_id=%s turn_order=%s duration_seconds=%s",
                session.answer_id,
                turn_order,
                duration_seconds,
            )
            # Đánh dấu TRƯỚC khi thoát: từ đây trở đi lượt này chắc chắn không còn audio nào tới
            # nữa, nên client không được chờ thêm vì nó. Xem _turns_without_audio.
            self._turns_without_audio.add((session.answer_id, turn_order))
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
                session_id=self.exam_attempt_id,
                # Đường WS này chỉ biết mỗi exam_attempt_id: nó không hề tham gia bắt tay WebRTC
                # nên không cầm candidate id lẫn stream id. Để RỖNG chứ không nhét exam_attempt_id
                # vào -- vox-streaming tra bù được từ session registry của nó, còn một id sai thì nó
                # không phân biệt nổi với id thật (xem docstring của push_alert).
                participant_id="",
                stream_id="",
                alert_type="WINDOW_FOCUS_LOST",
                confidence=1.0,
            )
        )

    def _handle_asset_playback_failed(self, message: dict) -> None:
        """Tài nguyên audio/video của câu hỏi không phát được ở máy thí sinh, kể cả sau một lần
        thử lại (WPF ExamViewModel.NotifyQuestionAssetMediaFailed).

        Hệ quả cần biết: thí sinh sẽ được hỏi -- và bị chấm -- về đoạn ghi âm họ CHƯA TỪNG NGHE.
        Luồng thi vẫn đi tiếp có chủ ý (chặn lại thì hỏng cả buổi thi), nên chỗ này tồn tại để lần
        sau có người khiếu nại "em không nghe thấy gì" thì còn tra được.

        CỐ Ý không gọi push_alert như focus_lost/camera_signal_lost: kênh cảnh báo dành cho hành vi
        của thí sinh, còn đây là lỗi kỹ thuật. Trộn hai loại vào nhau là cách nhanh nhất khiến giám
        thị ngừng tin những cảnh báo thật.
        """
        logger.warning(
            "[attempt_connection] asset_playback_failed exam_attempt_id=%s question=%s reason=%s",
            self.exam_attempt_id,
            message.get("questionNumber"),
            message.get("reason"),
        )

    async def _handle_camera_signal_lost(self, message: dict) -> None:
        """Camera ngừng gửi khung hình quá ngưỡng (WPF CameraSignalGuard).

        Phát hiện nằm ở máy trạm chứ không ở vox-streaming, và đó là chủ ý: vox-streaming chỉ
        thấy "không có media" nên không phân biệt nổi camera bị rút với đường truyền của học
        viên chết. Máy trạm thì biết -- nó thấy khung hình đứng lại trong khi tiến trình vẫn
        khoẻ. Hai lời buộc tội rất khác nhau, gộp lại là bất công với học viên mạng kém.

        Như focus_lost, KHÔNG đi qua should_emit_alert/cooldown: việc gộp theo điều kiện kéo dài
        đã làm ở đầu WPF bằng hai ngưỡng (banner tại chỗ trước, cảnh báo sau), nên tới được đây
        nghĩa là đã qua bộ lọc rồi. Lọc thêm lần nữa chỉ làm mất cảnh báo.
        """
        never_delivered = bool(message.get("neverDelivered"))
        logger.warning(
            "[attempt_connection] camera_signal_lost exam_attempt_id=%s captured_at=%s never_delivered=%s",
            self.exam_attempt_id,
            message.get("capturedAt"),
            never_delivered,
        )
        detail = (
            "Camera chưa gửi được khung hình nào kể từ đầu phiên"
            if never_delivered
            else "Camera ngừng gửi khung hình"
        )
        spawn(
            push_alert(
                session_id=self.exam_attempt_id,
                # Rỗng chứ không nhét exam_attempt_id vào -- xem _handle_focus_lost.
                participant_id="",
                stream_id="",
                alert_type="CAMERA_SIGNAL_LOST",
                confidence=1.0,
                stream_type="camera",
                detail=detail,
            )
        )

    async def _handle_camera_signal_restored(self, message: dict) -> None:
        """Khung hình đã trở lại sau một lần mất ĐÃ được cảnh báo.

        Tồn tại để đóng KHOẢNG chứ không phải để báo tin vui. Với người chấm, "mất camera lúc
        10:32" gần như vô dụng: hai mươi giây hay suốt phần còn lại của bài thi là hai kết luận
        hoàn toàn khác nhau. INFO chứ không WARNING -- lúc này sự cố đã qua, không còn gì để
        giám thị can thiệp; nó chỉ cần nằm trong sổ cho người chấm.
        """
        outage_seconds = message.get("outageSeconds")
        logger.info(
            "[attempt_connection] camera_signal_restored exam_attempt_id=%s captured_at=%s outage=%ss",
            self.exam_attempt_id,
            message.get("capturedAt"),
            outage_seconds,
        )
        try:
            outage_text = f"{float(outage_seconds):.0f}s"
        except (TypeError, ValueError):
            # Client cũ hoặc payload méo. Vẫn phát cảnh báo: biết "đã phục hồi" mà không biết bao
            # lâu vẫn hơn hẳn việc để khoảng trống trong sổ không bao giờ được đóng lại.
            outage_text = "không rõ"
        spawn(
            push_alert(
                session_id=self.exam_attempt_id,
                participant_id="",
                stream_id="",
                alert_type="CAMERA_SIGNAL_RESTORED",
                confidence=1.0,
                stream_type="camera",
                detail=f"Camera có hình trở lại sau {outage_text}",
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
        else:
            next_prompt = result.prompt_to_speak

        # Resume KHÔNG được phát lại câu thí sinh đã nghe rồi.
        #
        # `recovered_decision` không phải câu do LLM sinh mới -- nó là quyết định ĐÃ LƯU của lượt
        # cuối (turn_processor.recover_pending, nhánh dự phòng). Nhánh cũ ở đây gọi `_speak(...)`
        # vô điều kiện, nên mỗi lần nối lại là phát lại nguyên văn câu cũ. Đo thật 2026-08-26: hai
        # lần nối lại trong CÙNG một giây -> cùng một câu hỏi phát hai lần, cách nhau chưa tới một
        # giây, chồng tiếng lên nhau.
        #
        # Hậu quả nặng hơn phần âm thanh: mỗi vòng như vậy đẩy `turn_order` lên, và ngạch lượt của
        # câu hỏi bị đốt hết (2 -> 3 -> 4 -> "Thank you for your answer. Let's move on.") trong khi
        # thí sinh CHƯA trả lời được câu nào. Mất điểm, không phải chỉ khó chịu.
        if next_prompt and next_prompt == self._last_delivered_prompt:
            logger.info(
                "[attempt_connection] bo qua phat lai cau da gui exam_attempt_id=%s text=%r",
                self.exam_attempt_id,
                next_prompt,
            )
            return

        self._speak(next_prompt)

    def export_delivery_state(self) -> DeliveryState:
        """Ảnh chụp để kết nối thay thế nối tiếp, xem DeliveryState."""
        return DeliveryState(
            sequence=self._utterance_sequence,
            last_delivered_prompt=self._last_delivered_prompt,
        )

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
        # Chỉ ghi nhận SAU khi gửi trót lọt. `send_json` ném ClientDisconnected khi socket đã chết,
        # và câu đó chưa từng tới thí sinh -- đánh dấu ở đây thì lần resume sau sẽ im lặng bỏ qua
        # đúng câu cần phát lại. Đây là lý do dòng này nằm sau `await`, không phải trước.
        if text:
            self._last_delivered_prompt = text

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
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        self.questions.clear()
        await self.voice_live_client.close()
