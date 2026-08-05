"""Chặn lượt nói không phải tiếng Anh trước khi đem đi sửa lỗi.

Vì sao phải có RIÊNG ở đây, không dùng lại utils.speech_client.compute_code_switching_ratio:
hàm đó chỉ đếm chữ nằm trong markup "[XX: ...]", mà markup ấy do Azure Speech SDK gắn và chỉ
có ở đường THI (/turns/archive). Đường LUYỆN dùng transcript realtime của Voice Live -- chữ
trần, không bao giờ có thẻ -- nên hàm đó luôn trả 0.0 ở đây, kể cả với câu 100% tiếng Việt.

Sửa thẳng vào hàm dùng chung thì đổi luôn hành vi chấm THI ở những ca Azure gắn thẻ sót, nên
bộ dò này nằm riêng trong phạm vi luyện tập và không có ai bên thi gọi tới.

Hậu quả thật khi thiếu nó, đo được trên dữ liệu: lượt "Ý là tôi không biết câu này." được
- chấm bình thường (item_score 72.65, CAO HƠN lượt tiếng Anh thật 71.16),
- sửa NGỮ PHÁP TIẾNG VIỆT ("Ý tôi là..." kèm giải thích bằng tiếng Anh),
- và Azure chấm từ tiếng Việt bằng mô hình âm tiếng Anh nên trả 0% -> sinh phoneme_n,
  phoneme_k GIẢ, chạy thẳng vào hồ sơ điểm yếu rồi vào cả việc chọn câu hỏi kế tiếp.
"""

import re
from typing import Optional

# Ký tự CHỈ xuất hiện trong tiếng Việt. Tiếng Anh không dùng, nên một từ chứa bất kỳ ký tự nào
# trong đây gần như chắc chắn là tiếng Việt.
_VIETNAMESE_CHARS = (
    "ăâđêôơư"
    "àáảãạằắẳẵặầấẩẫậ"
    "èéẻẽẹềếểễệ"
    "ìíỉĩị"
    "òóỏõọồốổỗộờớởỡợ"
    "ùúủũụừứửữự"
    "ỳýỷỹỵ"
)
_VIETNAMESE_CHAR_SET = frozenset(_VIETNAMESE_CHARS + _VIETNAMESE_CHARS.upper())

_WORD_PATTERN = re.compile(r"[\wÀ-ỹ]+")

# Từ ngưỡng này trở lên thì thôi sửa lỗi. Lấy đúng ngưỡng đường thi đang dùng cho
# codeSwitchingRatio (mappers/exam_event_builder.py) -- luyện tập không nên dễ dãi hơn thi.
#
# Dưới ngưỡng (lẫn một hai từ tiếng Việt giữa câu tiếng Anh) vẫn sửa như thường: đó là
# code-switching bình thường của người học, chặn hết là phạt oan.
WRONG_LANGUAGE_RATIO = 0.5


def vietnamese_word_ratio(text: Optional[str]) -> float:
    """Tỉ lệ từ tiếng Việt trên tổng số từ. 0.0 khi không có chữ nào.

    Chỉ dựa vào DẤU PHỤ. STT trả về tiếng Việt có dấu nên đây là dấu hiệu gần như không thể
    nhầm. Tiếng Việt không dấu sẽ lọt -- chấp nhận có ý: mở rộng sang danh sách từ chức năng
    không dấu sẽ bắt nhầm cả tiếng Anh, mà chặn oan một câu tiếng Anh đúng thì tệ hơn bỏ sót.
    """
    words = _WORD_PATTERN.findall(text or "")
    if not words:
        return 0.0
    vietnamese = sum(
        1 for word in words if any(character in _VIETNAMESE_CHAR_SET for character in word)
    )
    return round(vietnamese / len(words), 2)


def is_wrong_language(text: Optional[str]) -> bool:
    return vietnamese_word_ratio(text) >= WRONG_LANGUAGE_RATIO


def is_vietnamese_word(word: Optional[str]) -> bool:
    """Một từ đơn có phải tiếng Việt không -- dùng để lọc kết quả chấm phát âm.

    Cần cho lượt LẪN ngôn ngữ (dưới WRONG_LANGUAGE_RATIO nên vẫn đi đường sửa lỗi bình
    thường): Azure chấm từng từ bằng mô hình âm tiếng Anh, nên mọi từ tiếng Việt trong đó đều
    trả accuracy gần 0 và bị báo là "phát âm sai". Học sinh nói đúng từ tiếng Việt của mình mà
    bị chấm sai, rồi lỗi giả đó thành phoneme_* trong hồ sơ điểm yếu.

    Lọc ở tầng KẾT QUẢ chứ không bỏ từ trước khi gọi Azure: âm thanh vẫn chứa cả câu, cắt chữ
    ra khỏi văn bản tham chiếu không làm Azure thôi nghe thấy tiếng Việt.
    """
    if not word:
        return False
    return any(character in _VIETNAMESE_CHAR_SET for character in word)
