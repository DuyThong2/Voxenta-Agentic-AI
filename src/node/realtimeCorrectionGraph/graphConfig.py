from langgraph.graph import END, START, StateGraph

from node.realtimeCorrectionGraph.GraphState import RealtimeCorrectionGraphState
from node.realtimeCorrectionGraph.LightCorrectionNode.light_correction_node_config import (
    light_correction_node,
)
from node.realtimeCorrectionGraph.PronunciationNode.pronunciation_node_config import (
    pronunciation_node,
)
from node.realtimeCorrectionGraph.WordChoiceNode.word_choice_node_config import (
    word_choice_node,
)

# Ngưỡng coi một từ là phát âm chưa đạt. Azure trả accuracy 0..100; trên mức này đa phần là
# nhiễu của bộ chấm chứ không phải lỗi thật, lôi ra chỉ làm học sinh mất tự tin vô cớ.
_PRONUNCIATION_WEAK_THRESHOLD = 70.0

# Tối đa 2 từ phát âm kém mỗi lượt -- cùng lý do với MAX_SUGGESTIONS bên WordChoiceNode:
# đây là thứ đọc lướt giữa hai lượt nói, không phải bảng điểm.
_MAX_PRONUNCIATION_ROWS = 2


def merge_correction_node(state: RealtimeCorrectionGraphState) -> dict:
    # NOTE: pronunciation_node, light_correction_node và word_choice_node chạy SONG SONG,
    # fan-out thẳng từ START (theo đúng khuôn prepare_turn_signals -> repeat_recovery +
    # followup_decision của followUpDecisionGraph). Chỉ trả về những khoá node này thực sự
    # đặt -- KHÔNG spread `state` ra lại, nếu không các kênh LastValue dùng chung của các
    # nhánh song song (transcript/audio_path/language) sẽ đụng nhau trong cùng một
    # superstep, đúng cảnh báo ở NOTE trong followup_decision_node_config.py.
    return {
        "corrections": state.get("light_corrections") or [],
        "status": "completed",
        "error": None,
    }


def _pronunciation_rows(state: RealtimeCorrectionGraphState) -> list:
    """Đổi kết quả chấm phát âm thành các dòng sửa cùng hình dạng với lỗi ngữ pháp.

    Đọc đúng MỘT hình dạng: `pronunciation_result["words"]`, là WordFeedback.model_dump() do
    PronunciationNode trả về (snake_case). Bản trước chấp nhận cả `words` lẫn `wordFeedback`
    và cả hai kiểu viết của mỗi trường -- kiểu "phòng xa" đó lại chính là thứ giấu lỗi: node
    khi ấy KHÔNG hề trả về từ nào cả, mà hàm này vẫn im lặng trả danh sách rỗng thay vì lộ ra.
    """
    result = state.get("pronunciation_result") or {}
    words = result.get("words") or []
    rows = []
    for word in words:
        if not isinstance(word, dict):
            continue
        score = word.get("accuracy_score")
        if score is None or float(score) >= _PRONUNCIATION_WEAK_THRESHOLD:
            continue
        # Âm vị kém nhất trong từ -- chính là "âm nào sai" cần chỉ mặt.
        worst = None
        for item in word.get("phonemes") or []:
            if not isinstance(item, dict):
                continue
            value = item.get("accuracy_score")
            if value is None:
                continue
            if worst is None or float(value) < float(worst[1]):
                worst = (item.get("phoneme") or "", value)
        # Nhét âm sai vào NGAY câu giải thích, không chỉ để ở trường riêng: bảng turn_correction
        # bên Java chỉ có category/original/corrected/explanation, nên khi mở lại buổi cũ thì
        # worst_phoneme không còn chỗ nào để đọc ra. Viết vào explanation là chỗ duy nhất nó
        # sống sót qua vòng lưu mà không phải đổi lược đồ.
        note = word.get("error_note") or "Phát âm chưa rõ."
        if worst and worst[0]:
            note = f"{note} Âm /{worst[0]}/ yếu nhất ({float(worst[1]):.0f}%)."
        rows.append(
            {
                "category": "pronunciation",
                "original_text": word.get("word") or "",
                "corrected_text": word.get("word") or "",
                "explanation": note,
                # Azure cho SỐ ĐO chứ không phán đoán, nên không có "độ tin cậy" tự nhiên nào.
                # Đặt 1.0 để nói rõ: đây là dữ liệu đo được, đừng lọc nó như lọc phán đoán LLM.
                "confidence": 1.0,
                "accuracy": float(score),
                "worst_phoneme": worst[0] if worst else None,
                "worst_phoneme_accuracy": float(worst[1]) if worst else None,
            }
        )
    rows.sort(key=lambda row: row["accuracy"])
    return rows[:_MAX_PRONUNCIATION_ROWS]


def format_feedback_node(state: RealtimeCorrectionGraphState) -> dict:
    """Node tổng kết: gộp ba nguồn thành MỘT danh sách đã phân loại và xếp thứ tự.

    Đặt việc này ở server chứ không để client tự ghép, vì ba lý do:
      - thứ tự ưu tiên (lỗi sai trước, gợi ý nâng cấp sau) là quyết định sư phạm chứ không
        phải chuyện trình bày -- để mỗi client tự nghĩ một kiểu thì chỗ nào cũng khác nhau;
      - `category_counts` cho client dựng tab lọc mà không phải đếm lại;
      - thêm client mới (web, WPF) là dùng được ngay, không phải chép lại logic ghép.
    """
    corrections = list(state.get("corrections") or [])

    # Gợi ý dùng từ hay hơn -> cùng hình dạng với các dòng sửa, nhưng gắn cờ `is_upgrade` để
    # client hiển thị khác đi: đây là NÂNG CẤP, gạch chân đỏ vào là sai thông điệp.
    for item in state.get("word_choices") or []:
        corrections.append(
            {
                "category": "vocabulary",
                "original_text": item.get("original_text") or "",
                "corrected_text": item.get("suggested_text") or "",
                "explanation": item.get("reason") or "",
                "is_upgrade": True,
            }
        )

    corrections.extend(_pronunciation_rows(state))

    # Lỗi thật lên trước, nâng cấp xuống cuối: học sinh chỉ liếc vài giây, thứ phải sửa
    # không được nằm dưới thứ tuỳ chọn.
    order = {"grammar": 0, "coherence": 1, "pronunciation": 2, "vocabulary": 3}
    corrections.sort(
        key=lambda row: (
            1 if row.get("is_upgrade") else 0,
            order.get(row.get("category"), 9),
        )
    )

    counts: dict = {}
    for row in corrections:
        key = row.get("category") or "other"
        counts[key] = counts.get(key, 0) + 1

    return {"corrections": corrections, "category_counts": counts}


def build_realtime_correction_graph():
    """Stateless per-call graph (no checkpointer) -- the caller (PracticeAttemptConnection)
    already has the transcript/audio_path in hand for this turn and doesn't need resume
    semantics for a feedback pass that either lands or doesn't."""
    g = StateGraph(RealtimeCorrectionGraphState)
    g.add_node("pronunciation", pronunciation_node)
    g.add_node("light_correction", light_correction_node)
    g.add_node("word_choice", word_choice_node)
    g.add_node("merge_correction", merge_correction_node)
    g.add_node("format_feedback", format_feedback_node)

    # Ba nhánh song song -> gộp -> định dạng. Thêm word_choice KHÔNG làm chậm thêm vì nó
    # chạy cùng lúc với hai nhánh kia; tổng thời gian vẫn bằng nhánh chậm nhất.
    g.add_edge(START, "pronunciation")
    g.add_edge(START, "light_correction")
    g.add_edge(START, "word_choice")
    g.add_edge("pronunciation", "merge_correction")
    g.add_edge("light_correction", "merge_correction")
    g.add_edge("word_choice", "merge_correction")
    g.add_edge("merge_correction", "format_feedback")
    g.add_edge("format_feedback", END)

    return g.compile()
