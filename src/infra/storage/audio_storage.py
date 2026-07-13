import asyncio
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from config.s3_config import settings

logger = logging.getLogger(__name__)


def _normalize_key(key: str) -> str:
    normalized = key.lstrip("/")
    prefix = (settings.AWS_S3_KEY_PREFIX or "").strip("/")
    if not prefix:
        return normalized
    if normalized == prefix or normalized.startswith(f"{prefix}/"):
        return normalized
    return f"{prefix}/{normalized}" if normalized else prefix


def _is_remote_ref(ref: str) -> bool:
    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https", "s3"}:
        return True
    return not Path(ref).exists()


def _resolve_bucket_and_key(ref: str) -> tuple[str, str]:
    parsed = urlparse(ref)

    if parsed.scheme == "s3":
        return parsed.netloc, _normalize_key(parsed.path.lstrip("/"))

    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.split(":")[0]
        path_parts = [part for part in parsed.path.split("/") if part]

        if ".s3." in host or host.endswith(".s3.amazonaws.com"):
            return host.split(".")[0], _normalize_key("/".join(path_parts))

        if path_parts and (
            host.startswith("s3.")
            or host == "s3.amazonaws.com"
            or settings.S3_ENDPOINT_URL
        ):
            bucket = path_parts[0]
            key = _normalize_key("/".join(path_parts[1:]))
            if bucket and key:
                return bucket, key

        if settings.S3_BUCKET_NAME:
            return settings.S3_BUCKET_NAME, _normalize_key(parsed.path.lstrip("/"))

        raise ValueError(f"Cannot infer S3 bucket from URL: {ref}")

    if settings.S3_BUCKET_NAME:
        return settings.S3_BUCKET_NAME, _normalize_key(ref)

    raise FileNotFoundError(f"Audio reference not found: {ref}")


def download_from_s3(ref: str) -> str:
    if not _is_remote_ref(ref):
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"Audio reference not found: {ref}")
        return str(path)

    bucket, key = _resolve_bucket_and_key(ref)
    if not bucket or not key:
        raise ValueError(f"Invalid S3 audio reference: {ref}")

    suffix = Path(key).suffix or ".wav"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.close()

    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL or None,
            config=Config(
                s3={
                    "addressing_style": "path" if settings.AWS_S3_PATH_STYLE else "virtual",
                }
            ),
        )
        client.download_file(bucket, key, tmp_file.name)
        logger.info("Downloaded audio ref=%s bucket=%s key=%s", ref, bucket, key)
        return tmp_file.name
    except Exception:
        Path(tmp_file.name).unlink(missing_ok=True)
        raise


async def download_from_s3_async(ref: str) -> str:
    return await asyncio.to_thread(download_from_s3, ref)


def upload_to_s3(local_path: str) -> str:
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    raise NotImplementedError("S3 upload helper is not implemented yet")
