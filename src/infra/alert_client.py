import logging
import os
import time

import grpc

from infra.grpc_stubs.alert.v1 import alert_pb2, alert_pb2_grpc

logger = logging.getLogger(__name__)


async def push_alert(
    room_id: str,
    participant_id: str,
    stream_id: str,
    alert_type: str,
    confidence: float = 1.0,
) -> None:
    grpc_addr = str(os.getenv("VOX_STREAMING_GRPC_ADDR", "") or "").strip()
    api_key = str(os.getenv("VOX_STREAMING_API_KEY", "") or "").strip()
    if not grpc_addr:
        logger.warning("[alert_client] VOX_STREAMING_GRPC_ADDR not configured, skipping alert %s", alert_type)
        return

    channel = grpc.aio.insecure_channel(grpc_addr)
    try:
        stub = alert_pb2_grpc.AlertServiceStub(channel)
        request = alert_pb2.PushAlertRequest(
            roomId=room_id,
            participantId=participant_id,
            streamId=stream_id,
            alertType=alert_type,
            confidence=float(confidence),
            capturedAtMs=int(time.time() * 1000),
        )
        metadata = (("authorization", f"Bearer {api_key}"),) if api_key else None
        response = await stub.PushAlert(request, metadata=metadata, timeout=5.0)
        logger.info(
            "[alert_client] pushed alert_type=%s room=%s participant=%s received=%s",
            alert_type,
            room_id,
            participant_id,
            getattr(response, "received", False),
        )
    except Exception:
        logger.exception("[alert_client] failed to push alert_type=%s room=%s", alert_type, room_id)
    finally:
        await channel.close()
