"""Lexical resource evaluation node using LLM to assess vocabulary range and accuracy."""

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.LexicalEvalNode.lexical_eval_prompt import SYSTEM_PROMPT


def build_user_prompt(transcript: str, reference_text: str | None, mode: str) -> str:
    parts = [f"Mode: {mode}", f'Transcript: "{transcript}"']

    if reference_text:
        parts.append(f'Reference text: "{reference_text}"')

    if mode == "scripted":
        parts.append("\nThe speaker was reading the reference text aloud. Evaluate lexical resource based on word accuracy against the reference.")
    else:
        parts.append("\nThe speaker was speaking freely. Evaluate lexical resource based on vocabulary diversity and appropriateness.")

    return "\n".join(parts)


def call_llm(transcript: str, reference_text: str | None, mode: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(transcript, reference_text, mode)),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def merge_lexical_score(pronunciation_result: Any, llm_response: Dict[str, Any]) -> Any:
    """Merge lexical resource score into pronunciation_result.criteria."""
    criteria = dict(pronunciation_result.criteria)

    criteria["lexical_resource"] = {
        "score": llm_response["score"],
        "subscores": llm_response.get("subscores", {}),
        "note": llm_response.get("note", "Evaluated by LLM based on transcript analysis."),
    }

    pronunciation_result.criteria = criteria
    return pronunciation_result


def lexical_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to evaluate lexical resource using LLM.

    Reads: speaking_input, pronunciation_result
    Updates: pronunciation_result.criteria.lexical_resource
    """

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for lexical_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    transcript = (
        speaking_input.corrected_transcript
        or speaking_input.transcribed_text
        or pronunciation_result.recognized_text
    )

    if not transcript:
        return {**state, "status": "error", "error": "No transcript available for lexical evaluation"}

    reference_text = speaking_input.reference_text
    mode = speaking_input.mode or "unscripted"

    try:
        llm_response = call_llm(transcript, reference_text, mode)
        updated_result = merge_lexical_score(pronunciation_result, llm_response)

        return {**state, "pronunciation_result": updated_result, "status": "processing", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"Lexical evaluation failed: {str(exc)}"}
