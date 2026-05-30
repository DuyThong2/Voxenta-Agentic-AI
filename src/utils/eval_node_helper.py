"""Shared helper for LLM-based evaluation nodes.

Eliminates duplication across CoherenceEvalNode, LexicalEvalNode, GrammarEvalNode.
Each node only needs to provide: criterion_key, SYSTEM_PROMPT, and build_user_prompt().
"""

import json
from typing import Any, Callable, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.state_models import SpeakingInput
from schemas.scoring import CriterionScore
from utils.transcript_selector import select_text_for_language_scoring, build_scoring_metadata


def call_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON response. Shared by all eval nodes."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def merge_criterion(pronunciation_result: Any, criterion_key: str, llm_response: Dict[str, Any]) -> Any:
    """Merge LLM score into pronunciation_result.criteria.<criterion_key>."""
    criterion = CriterionScore(
        score=llm_response["score"],
        subscores=llm_response.get("subscores", {}),
        note=llm_response.get("note", "Evaluated by LLM based on transcript analysis."),
    )
    setattr(pronunciation_result.criteria, criterion_key, criterion)
    return pronunciation_result


def run_eval_node(
    state: Dict[str, Any],
    criterion_key: str,
    system_prompt: str,
    build_user_prompt_fn: Callable[[SpeakingInput, str], str],
    node_name: str,
) -> Dict[str, Any]:
    """Generic eval node runner. Handles guards, transcript selection, LLM call, merge, error handling."""
    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": f"speaking_input is required for {node_name}_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    transcript, source = select_text_for_language_scoring(speaking_input)

    if transcript is None:
        return {**state, "status": "error", "error": f"No transcript available for {node_name} evaluation"}

    scoring_meta = build_scoring_metadata(source, speaking_input.mode)

    try:
        user_prompt = build_user_prompt_fn(speaking_input, transcript)
        llm_response = call_llm(system_prompt, user_prompt)
        updated_result = merge_criterion(pronunciation_result, criterion_key, llm_response)

        existing_meta = state.get("metadata") or {}
        merged_meta = {**existing_meta, **scoring_meta}

        return {**state, "pronunciation_result": updated_result, "metadata": merged_meta, "status": "completed", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"{node_name} evaluation failed: {str(exc)}"}
