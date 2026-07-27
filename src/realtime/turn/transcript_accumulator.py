"""Per-question Voice Live transcript accumulation.

A speaking turn may contain multiple VAD-delimited utterances. This component owns
the in-flight partial/final state and returns one immutable snapshot when WPF ends
the turn. Consuming a turn resets both text and confidence so later turns cannot
inherit the previous turn's ASR reliability.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional


PENDING_TRANSCRIPT_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class TranscriptSnapshot:
    text: str
    confidence: Optional[float]


class TranscriptAccumulator:
    def __init__(self) -> None:
        self._finalized_text = ""
        self._live_partial = ""
        self._awaiting_final = False
        self._final_event = asyncio.Event()
        self._confidence_weighted_sum = 0.0
        self._confidence_weight_total = 0

    def on_speech_start(self) -> None:
        self._live_partial = ""

    def on_partial_transcript(self, text: Optional[str]) -> None:
        if text:
            self._live_partial += text

    def on_speech_end(self) -> None:
        self._awaiting_final = True
        self._final_event.clear()

    def on_final_transcript(
        self,
        text: Optional[str],
        confidence: Optional[float] = None,
    ) -> None:
        finalized = (text or self._live_partial or "").strip()
        if finalized:
            self._finalized_text = f"{self._finalized_text} {finalized}".strip()
            if confidence is not None:
                weight = max(len(finalized.split()), 1)
                self._confidence_weighted_sum += confidence * weight
                self._confidence_weight_total += weight

        self._live_partial = ""
        self._awaiting_final = False
        self._final_event.set()

    async def consume_turn(
        self,
        timeout: float = PENDING_TRANSCRIPT_TIMEOUT_SECONDS,
    ) -> TranscriptSnapshot:
        if self._awaiting_final:
            try:
                await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                if self._live_partial.strip():
                    self._finalized_text = (
                        f"{self._finalized_text} {self._live_partial.strip()}".strip()
                    )
                self._live_partial = ""
                self._awaiting_final = False

        confidence = (
            self._confidence_weighted_sum / self._confidence_weight_total
            if self._confidence_weight_total > 0
            else None
        )
        snapshot = TranscriptSnapshot(self._finalized_text, confidence)
        self.reset()
        return snapshot

    def reset(self) -> None:
        self._finalized_text = ""
        self._live_partial = ""
        self._awaiting_final = False
        self._final_event.set()
        self._confidence_weighted_sum = 0.0
        self._confidence_weight_total = 0
