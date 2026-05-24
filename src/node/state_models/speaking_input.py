from typing import Literal, Optional

from pydantic import BaseModel


class SpeakingInput(BaseModel):
    audio_path: str

    # Có reference_text thì scripted.
    # Không có reference_text thì unscripted.
    reference_text: Optional[str] = None

    mode: Literal["scripted", "unscripted"] = "unscripted"
    language: str = "en-US"