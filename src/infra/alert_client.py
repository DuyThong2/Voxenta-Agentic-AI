import logging
import os
import time

import grpc

from infra.grpc_stubs.alert.v1 import alert_pb2, alert_pb2_grpc

logger = logging.getLogger(__name__)


async def push_alert(
    session_id: str,
    participant_id: str,
    stream_id: str,
    alert_type: str,
    confidence: float = 1.0,
    stream_type: str = "",
    detail: str = "",
) -> None:
    """Đẩy một cảnh báo giám sát sang vox-streaming.

    Ba định danh là ba thứ KHÁC NHAU và không được phép thay thế cho nhau:

    - ``session_id``: phiên thi (exam attempt). Đây là khoá vox-streaming dùng để định tuyến cảnh báo
      tới đúng màn hình giám thị đang mở.
    - ``participant_id``: thí sinh (candidate id). Đây là khoá giao diện dùng để tra ra TÊN học viên
      và gắn cảnh báo vào đúng ô trên lưới.
    - ``stream_id``: luồng cụ thể sinh ra cảnh báo.

    Trước đây cả ba đều bị gán bằng ``session_id``. Hậu quả không dừng ở chỗ dòng cảnh báo in ra một
    chuỗi UUID thay cho tên: khoá sai khiến cảnh báo không khớp ô nào, nên trạng thái "Có cảnh báo",
    viền đỏ và thứ tự ưu tiên của lưới đều ngừng hoạt động với mọi cảnh báo do AI sinh.

    Không biết thì để rỗng, đừng lấy id khác điền vào. vox-streaming tự tra lại được phần thiếu từ
    session registry của nó (nó là bên duy nhất từng biết ánh xạ này, vì nó đúc peer từ stream token
    của học viên), nhưng nó không có cách nào phát hiện một id sai đã được điền tự tin vào đó.
    """
    grpc_addr = str(os.getenv("VOX_STREAMING_GRPC_ADDR", "") or "").strip()
    api_key = str(os.getenv("VOX_STREAMING_API_KEY", "") or "").strip()
    if not grpc_addr:
        logger.warning("[alert_client] VOX_STREAMING_GRPC_ADDR not configured, skipping alert %s", alert_type)
        return

    channel = grpc.aio.insecure_channel(grpc_addr)
    try:
        stub = alert_pb2_grpc.AlertServiceStub(channel)
        request = alert_pb2.PushAlertRequest(
            session_id=session_id,
            participant_id=participant_id,
            stream_id=stream_id,
            alert_type=alert_type,
            confidence=float(confidence),
            captured_at_ms=int(time.time() * 1000),
            stream_type=stream_type,
            detail=detail,
        )
        metadata = (("authorization", f"Bearer {api_key}"),) if api_key else None
        response = await stub.PushAlert(request, metadata=metadata, timeout=5.0)
        logger.info(
            "[alert_client] pushed alert_type=%s session=%s participant=%s received=%s",
            alert_type,
            session_id,
            participant_id or "(unknown)",
            getattr(response, "received", False),
        )
    except Exception:
        logger.exception("[alert_client] failed to push alert_type=%s session=%s", alert_type, session_id)
    finally:
        await channel.close()
