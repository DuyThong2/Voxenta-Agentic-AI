"""Fan-in node for pronunciation and combined language-quality scoring.

Concurrent branches only write namespaced metadata. This node is the single
place that sets the shared status/error and attaches language criteria to the
pronunciation result.
"""

import logging
from typing import Any, Dict

from node.state_models.pronunciation import FormattedPronunciationResult

logger = logging.getLogger(__name__)

_BRANCH_ERROR_KEYS = (
    "pronunciation_error",
    "azure_score_scale_error",
    "answer_length_error",
    "language_quality_error",
)

# Lỗi của riêng nhánh phát âm. Khi lượt nói KHÔNG có audio thì cả ba thứ dưới đây đều là hệ quả
# của đúng một việc lành tính, không phải ba sự cố:
#   - pronunciation_error     : "speaking_input.audio_path is required"
#   - azure_score_scale_error : "speaking_input and pronunciation_result are required"
#   - pronunciation_result    : None
_PRONUNCIATION_ERROR_KEYS = ("pronunciation_error", "azure_score_scale_error")

_NOT_ASSESSED_NOTE = (
    "Không chấm phát âm cho lượt này: không thu được bản ghi âm. "
    "Ngữ pháp, từ vựng và mạch lạc vẫn được chấm từ bản ghi lời nói."
)


def merge_scores_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = state.get("metadata") or {}
    branch_errors = {key: metadata[key] for key in _BRANCH_ERROR_KEYS if metadata.get(key)}

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    # Thiếu audio là chuyện BÌNH THƯỜNG, không phải sự cố: bên thi giữ lại lượt nói dù bản ghi âm
    # không tới được S3, nên exam_item_response_turns.audio_url có thể rỗng.
    #
    # Phân biệt bằng chính audio_path chứ không bằng nội dung thông báo lỗi: không có đường dẫn
    # nghĩa là KHÔNG CÓ CƠ SỞ để chấm phát âm; có đường dẫn mà vẫn lỗi nghĩa là Azure hoặc file
    # thật sự có vấn đề -- cái sau vẫn phải là lỗi, không được nuốt.
    #
    # Trước bản này, một lượt im lặng làm merge trả status="error", _evaluate_turn ném RuntimeError,
    # cả 4 lần retry hỏng y hệt rồi bài bị đánh ExamAttemptEvaluationFailed -- mất điểm luôn những
    # lượt đã chấm xong hoàn hảo (đo được 2026-08-18, phiên 01a015a8: 3 lượt có điểm đầy đủ, 1 lượt
    # im lặng, kết quả là cả 4 mất trắng).
    audio_absent = speaking_input is None or not getattr(speaking_input, "audio_path", None)

    if audio_absent:
        dropped = [key for key in _PRONUNCIATION_ERROR_KEYS if key in branch_errors]
        for key in dropped:
            branch_errors.pop(key)
        if dropped or pronunciation_result is None:
            logger.warning(
                "[eval:merge_scores] khong co audio -> bo qua phat am, van cham ngu lieu "
                "answer_id=%s turn=%s bo_qua=%s",
                getattr(speaking_input, "answer_id", None),
                metadata.get("turn_order"),
                dropped,
            )
        if pronunciation_result is None:
            # Vỏ rỗng để còn chỗ gắn điểm ngữ pháp/từ vựng/mạch lạc. Mọi field điểm phát âm để
            # trống -- KHÔNG phải 0, vì 0 sẽ đi vào bảng điểm như một con điểm thật.
            #
            # criteria.pronunciation rỗng nên phía Java không thấy tiêu chí PRONUNCIATION và bỏ nó
            # khỏi CẢ tử số lẫn mẫu số (RubricItemScoreFormula cộng weightSum theo tiêu chí thật sự
            # có mặt), nên điểm câu giữ đúng tỉ lệ thay vì bị chia hụt phần trọng số phát âm.
            pronunciation_result = FormattedPronunciationResult(notes=[_NOT_ASSESSED_NOTE])
        elif _NOT_ASSESSED_NOTE not in pronunciation_result.notes:
            pronunciation_result.notes.append(_NOT_ASSESSED_NOTE)
    elif pronunciation_result is None:
        branch_errors.setdefault("pronunciation_error", "pronunciation_result missing at merge_scores")

    if branch_errors:
        combined = "; ".join(f"{key}: {message}" for key, message in branch_errors.items())
        return {"status": "error", "error": combined}

    coherence_criterion = state.get("coherence_criterion")
    lexical_criterion = state.get("lexical_criterion")
    grammar_criterion = state.get("grammar_criterion")

    if coherence_criterion is not None:
        pronunciation_result.criteria.coherence = coherence_criterion
    if lexical_criterion is not None:
        pronunciation_result.criteria.vocabulary = lexical_criterion
    if grammar_criterion is not None:
        pronunciation_result.criteria.grammar = grammar_criterion

    return {
        "pronunciation_result": pronunciation_result,
        "status": "completed",
        "error": None,
    }
