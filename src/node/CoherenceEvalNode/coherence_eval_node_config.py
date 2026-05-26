"""Coherence evaluation node using LLM to assess logical flow and connected discourse."""

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.CoherenceEvalNode.coherence_eval_prompt import SYSTEM_PROMPT


def build_user_prompt(transcript: str, reference_text: str | None, mode: str) -> str:
    parts = [f"Mode: {mode}", f'Transcript: "{transcript}"']

    if reference_text:
        parts.append(f'Reference text: "{reference_text}"')

    if mode == "scripted":
        parts.append("\nThe speaker was reading the reference text aloud. Evaluate coherence based on natural flow and rhythm.")
    else:
        parts.append("\nThe speaker was speaking freely. Evaluate coherence based on logical structure and connected discourse.")

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


def merge_coherence_score(pronunciation_result: Any, llm_response: Dict[str, Any]) -> Any:
    """Merge coherence score into pronunciation_result.criteria.fluency_and_coherence."""
    criteria = dict(pronunciation_result.criteria)

    fc = dict(criteria.get("fluency_and_coherence", {}))
    subscores = dict(fc.get("subscores", {}))

    coherence_score = llm_response["score"]
    subscores["coherence"] = coherence_score

    fluency_score = subscores.get("fluency")
    if fluency_score is not None:
        fc["score"] = round((fluency_score + coherence_score) / 2, 1)

    fc["subscores"] = subscores
    fc["note"] = "Fluency from audio assessment. Coherence from LLM transcript analysis."
    criteria["fluency_and_coherence"] = fc

    pronunciation_result.criteria = criteria
    return pronunciation_result


def coherence_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to evaluate coherence using LLM.

    Reads: speaking_input, pronunciation_result
    Updates: pronunciation_result.criteria.fluency_and_coherence.coherence
    """

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for coherence_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    transcript = (
        speaking_input.corrected_transcript
        or speaking_input.transcribed_text
        or pronunciation_result.recognized_text
    )

    if not transcript:
        return {**state, "status": "error", "error": "No transcript available for coherence evaluation"}

    reference_text = speaking_input.reference_text
    mode = speaking_input.mode or "unscripted"

    try:
        llm_response = call_llm(transcript, reference_text, mode)
        updated_result = merge_coherence_score(pronunciation_result, llm_response)

        return {**state, "pronunciation_result": updated_result, "status": "processing", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"Coherence evaluation failed: {str(exc)}"}
