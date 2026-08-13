from typing import Any, Dict, List, Optional, TypedDict


class RealtimeCorrectionGraphState(TypedDict, total=False):
    # Input
    transcript: str
    audio_path: Optional[str]
    language: str
    # Khoá buffer cho ai_usage_tracker (xem infra/message_broker/ai_usage_tracker.py) -- các
    # node gọi LLM/Azure thật (LightCorrection/WordChoice/EnglishRendering/Pronunciation) ghi
    # usage vào đây để connection.py pop lại ngay sau khi graph chạy xong, tính turn_cost_usd.
    answer_id: Optional[str]

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

    # Written by wrong_language_node -- lượt nói không phải tiếng Anh, đã bỏ qua toàn bộ việc
    # sửa lỗi. Client dùng cờ này để nói cho học sinh biết vì sao lượt vừa rồi không có phản
    # hồi, thay vì hiện một thẻ trống trông như hệ thống hỏng.
    wrong_language: bool

    status: str
    error: Optional[str]
