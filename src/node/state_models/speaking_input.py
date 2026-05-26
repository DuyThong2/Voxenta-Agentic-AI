from typing import Literal, Optional

from pydantic import BaseModel


class SpeakingInput(BaseModel):
    audio_path: str

    # Có reference_text thì scripted.
    # Không có reference_text thì unscripted.
    # Trong scripted mode, reference_text là nguồn dữ liệu chính xác nhất.
    reference_text: Optional[str] = None

    # Raw transcript from the speech recognizer.
    # Start node ghi vào đây trước khi correction node xử lý.
    transcribed_text: Optional[str] = None

    # Corrected transcript after correction node.
    # Chỉ dùng khi không có reference_text (unscripted mode).
    corrected_transcript: Optional[str] = None

    mode: Literal["scripted", "unscripted"] = "unscripted"
    language: str = "en-US"