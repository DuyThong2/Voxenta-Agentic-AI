from pathlib import Path
from urllib.parse import urlparse


def download_from_s3(ref: str) -> str:
    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https", "s3"}:
        raise NotImplementedError("S3 download helper is not implemented yet")

    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Audio reference not found: {ref}")
    return str(path)


def upload_to_s3(local_path: str) -> str:
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    raise NotImplementedError("S3 upload helper is not implemented yet")
