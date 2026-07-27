"""WebRTC + YOLO proctoring configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from aiortc import RTCConfiguration, RTCIceServer
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class WebRtcSettings(BaseSettings):
    """WebRTC signaling + YOLO proctoring settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    WEBRTC_MAX_CONNECTIONS: int = 100
    STUN_URLS: str = "stun:stun.l.google.com:19302"
    TURN_URL: str = ""
    TURN_USERNAME: str = ""
    TURN_CREDENTIAL: str = ""
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_FRAME_SKIP: int = 10
    YOLO_CONFIDENCE: float = 0.5
    # Minimum seconds between repeated alerts of the same type for the same session, once that
    # alert type is already active -- without this, PERSON_MISSING/MULTIPLE_PERSONS fire on every
    # processed frame (every YOLO_FRAME_SKIP frames) for as long as the condition holds, flooding
    # SSE/logs/the gRPC alert push with duplicates instead of one alert per actual state change.
    PROCTORING_ALERT_COOLDOWN_SECONDS: float = 5
    # Consecutive processed frames a condition must hold before it's treated as real rather than
    # single-frame detector noise (e.g. YOLO confidence dipping just under YOLO_CONFIDENCE for one
    # frame while a person is genuinely on camera -- lighting/angle/motion blur). Without this,
    # should_emit_alert's edge-trigger fires immediately on every such blip, since each 0<->1 flap
    # looks like a fresh state change.
    PROCTORING_ALERT_STREAK_FRAMES: int = 3


@lru_cache
def get_webrtc_settings() -> WebRtcSettings:
    """Get cached WebRTC/proctoring settings instance."""
    return WebRtcSettings()


settings = get_webrtc_settings()


def build_ice_servers() -> RTCConfiguration:
    """Build aiortc ICE configuration from environment-backed settings."""
    stun_urls = [url.strip() for url in settings.STUN_URLS.split(",") if url.strip()]
    if not stun_urls:
        stun_urls = ["stun:stun.l.google.com:19302"]

    servers = [
        RTCIceServer(urls=stun_urls)
    ]

    turn_url = settings.TURN_URL.strip()
    if turn_url:
        servers.append(
            RTCIceServer(
                urls=[turn_url],
                username=settings.TURN_USERNAME,
                credential=settings.TURN_CREDENTIAL,
            )
        )

    return RTCConfiguration(iceServers=servers)
