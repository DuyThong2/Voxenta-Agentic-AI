from typing import Any, Dict, List, Optional, TypedDict


class RealtimeCorrectionGraphState(TypedDict, total=False):
    # Input
    transcript: str
    audio_path: Optional[str]
    language: str

    # Written by pronunciation_node -- None if audio_path wasn't available (see node
    # docstring: turn-level PCM buffering-to-WAV isn't wired up yet, known gap).
    pronunciation_result: Optional[Dict[str, Any]]

    # Written by light_correction_node
    light_corrections: Optional[List[Dict[str, Any]]]

    # Written by merge_correction_node -- final TurnCorrection-shaped list, what
    # PracticeAttemptConnection pushes to the client / forwards to Java's
    # /internal/practice-sessions/{id}/turns
    corrections: List[Dict[str, Any]]
    status: str
    error: Optional[str]
