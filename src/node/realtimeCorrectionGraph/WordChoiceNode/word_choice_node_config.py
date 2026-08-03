"""Gợi ý dùng từ hay hơn -- chạy SONG SONG với light_correction, không nối tiếp.

Vì sao tách thành node riêng thay vì nhét thêm vào prompt sửa lỗi: hai việc có tiêu chí ngược
nhau. Sửa lỗi được dặn "chỉ báo cái chắc chắn sai, bỏ qua thứ mơ hồ"; gợi ý dùng từ thì lại
nhắm đúng vào chỗ KHÔNG sai. Gộp một prompt là bắt LLM cân hai thước đo trái chiều trong một
lượt, và thực tế nó sẽ nghiêng hẳn về bắt lỗi rồi bỏ quên phần nâng cấp.

Chạy song song nên không cộng thêm độ trễ: tổng thời gian vẫn bằng nhánh chậm nhất (thường là
pronunciation vì phải chấm audio), không phải tổng ba nhánh.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from node.realtimeCorrectionGraph.WordChoiceNode.word_choice_prompt import (
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Tối đa 2: học sinh đọc giữa hai lượt nói, danh sách dài là không ai đọc. Prompt cũng đã
# dặn thà trả rỗng còn hơn nhồi cho đủ.
MAX_SUGGESTIONS = 2


def _parse_json_array(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        lines = [
            line
            for line in content.splitlines()
            if not line.strip().startswith("```")
        ]
        content = "\n".join(lines).strip()
    parsed = json.loads(content)
    return parsed if isinstance(parsed, list) else []


def word_choice_node(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = (state.get("transcript") or "").strip()
    # Câu quá ngắn thì không có gì để nâng cấp, mà vẫn tốn một lượt gọi LLM.
    if len(transcript.split()) < 4:
        return {"word_choices": []}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Transcript:\n{transcript}"),
    ]
    try:
        response = llm.invoke(messages)
        suggestions = _parse_json_array(response.content)
    except Exception:
        # Nuốt lỗi có chủ đích: đây là phần "có thì tốt". Hỏng nhánh này không được kéo theo
        # phần sửa lỗi -- thứ học sinh thật sự cần -- chết theo.
        logger.exception("[realtime_correction:word_choice] failed")
        return {"word_choices": []}

    cleaned = [
        item
        for item in suggestions
        if isinstance(item, dict)
        and str(item.get("original_text") or "").strip()
        and str(item.get("suggested_text") or "").strip()
        # LLM đôi khi "gợi ý" đúng lại chính từ cũ -- bỏ, vì hiện ra thành mũi tên trỏ vào
        # chính nó, trông như lỗi hiển thị.
        and str(item.get("original_text")).strip().lower()
        != str(item.get("suggested_text")).strip().lower()
    ]
    return {"word_choices": cleaned[:MAX_SUGGESTIONS]}
