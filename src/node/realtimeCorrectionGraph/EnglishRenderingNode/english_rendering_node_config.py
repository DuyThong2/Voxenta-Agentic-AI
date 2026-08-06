"""Lượt nói bằng tiếng Việt -> đề xuất chính câu đó bằng tiếng Anh.

Thay cho việc chặn im lặng. Học sinh nói tiếng Việt hầu như luôn vì KHÔNG BIẾT nói thế nào
bằng tiếng Anh -- đúng lúc đó là lúc cần câu tiếng Anh nhất. Chặn rồi thôi là bỏ mất khoảnh
khắc dạy được duy nhất trong lượt đó.

Hai thứ node này KHÔNG làm, và đó mới là phần quan trọng:

1. Không sửa ngữ pháp tiếng Việt. Bản trước `light_correction_node` nhận nguyên câu tiếng Việt
   và trả về "Ý là..." -> "Ý tôi là..." kèm giải thích bằng tiếng Anh. Một ứng dụng luyện
   tiếng Anh đi dạy ngữ pháp tiếng Việt.

2. Không cho Azure chấm phát âm lượt này. Azure chấm từ tiếng Việt bằng mô hình âm tiếng Anh
   nên luôn trả gần 0%, sinh ra phoneme_n/phoneme_k GIẢ rồi chảy vào weakness_observation ->
   sub_attribute_priority -> hồ sơ điểm yếu -> và cả việc chọn câu hỏi kế tiếp. Một câu tiếng
   Việt đủ để đầu độc mô hình người học.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """The learner is practising spoken English but answered in Vietnamese,
almost always because they did not know how to say it in English.

Give them the English they were reaching for.

Return ONLY a JSON object:
{"english": "<the same meaning, said naturally in spoken English>",
 "note": "<one short sentence in Vietnamese telling them what to try>"}

Rules:
- Keep it at the level they were speaking at. Do not upgrade it into an essay sentence.
- Spoken register, contractions welcome. This is something they will say out loud next turn.
- Preserve their meaning exactly, including hesitation or not knowing an answer -- if they
  said they don't know, the English must also say they don't know. Do not invent an answer
  they did not give.
- If the Vietnamese is unintelligible, return an empty string for "english"."""


def english_rendering_node(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        return _blocked([])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Vietnamese:\n{transcript}"),
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.splitlines() if not line.strip().startswith("```")
            ).strip()
        parsed = json.loads(content)
    except Exception:
        # Hỏng thì vẫn phải chặn -- trả danh sách rỗng, KHÔNG rơi xuống nhánh sửa lỗi bình
        # thường. Mất gợi ý thì tiếc, còn để tiếng Việt lọt vào chuỗi chấm thì hỏng dữ liệu.
        logger.exception("[realtime_correction:english_rendering] failed")
        return _blocked([])

    english = (parsed.get("english") or "").strip()
    if not english:
        return _blocked([])

    return _blocked([
        {
            # Xếp vào "vocabulary" và gắn is_upgrade: đây là GỢI Ý cách nói, không phải lỗi
            # sai. Gạch chân đỏ vào câu tiếng Việt là nói sai thông điệp -- em ấy không sai,
            # em ấy chưa biết nói.
            "category": "vocabulary",
            "original_text": transcript,
            "corrected_text": english,
            "explanation": (parsed.get("note") or "").strip()
            or "Thử nói lại câu này bằng tiếng Anh.",
            "is_upgrade": True,
        }
    ])


def _blocked(corrections: list) -> Dict[str, Any]:
    return {
        "corrections": corrections,
        "category_counts": {"vocabulary": len(corrections)} if corrections else {},
        "pronunciation_result": None,
        "wrong_language": True,
        "status": "completed",
        "error": None,
    }
