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

    # Written by word_choice_node -- gợi ý dùng từ hay hơn, KHÔNG phải lỗi sai
    word_choices: Optional[List[Dict[str, Any]]]

    # Written by format_feedback_node -- số lượng theo từng loại, để client dựng tab lọc
    # mà không phải tự đếm lại
    category_counts: Dict[str, int]

    # Written by merge_correction_node -- final TurnCorrection-shaped list, what
    # PracticeAttemptConnection pushes to the client / forwards to Java's
    # /internal/practice-sessions/{id}/turns
    corrections: List[Dict[str, Any]]
    status: str
    error: Optional[str]
