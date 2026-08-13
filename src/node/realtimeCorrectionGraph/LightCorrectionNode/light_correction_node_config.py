"""1 LLM call, single round -- deliberately NOT the multi-round Drafter/Evaluator/Editor
pipeline questionGenerationGraph uses. This is realtime feedback shown while the learner reads
between turns, not a scored artifact; the official score still comes from evalGraph at the end
of the session (see task/implement/11-toi-uu-dung-de.md mục 2.4c)."""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from infra.message_broker import ai_usage_tracker
from node.realtimeCorrectionGraph.LightCorrectionNode.light_correction_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _parse_json_array(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
        content = "\n".join(lines).strip()
    parsed = json.loads(content)
    return parsed if isinstance(parsed, list) else []


def light_correction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        return {"light_corrections": []}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Transcript:\n{transcript}"),
    ]
    try:
        response = llm.invoke(messages)
        ai_usage_tracker.record_llm_usage(state.get("answer_id"), "openai", "gpt-4o-mini", response)
        corrections = _parse_json_array(response.content)
    except Exception:
        logger.exception("[realtime_correction:light_correction] failed")
        return {"light_corrections": []}

    return {"light_corrections": corrections[:3]}
