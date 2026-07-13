"""Script correction node using OpenAI to refine Azure transcription."""

import logging
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.evalGraph.CorrectionNode.correction_node_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def correct_transcript(transcript: str) -> str:
    """
    Use OpenAI to correct and refine Azure transcription.
    
    Args:
        transcript: Raw transcript from Azure Speech-to-Text
    
    Returns:
        Corrected transcript for better pronunciation assessment accuracy
    """
    if not transcript or not transcript.strip():
        return transcript
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Correct this transcription:\n\n{transcript}")
    ]
    
    response = llm.invoke(messages)
    corrected = response.content.strip()
    
    return corrected


def correction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to correct transcribed text using OpenAI.
    
    Expected state:
    {
        "speaking_input": SpeakingInput(...)
    }
    
    Returns state with:
    {
        "speaking_input": SpeakingInput updated with corrected_transcript,
        "status": "processing"
    }
    
    If reference_text is already present, this node is skipped because the
    final pronunciation node compares audio directly against reference_text.
    """
    
    speaking_input = state.get("speaking_input")
    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")

    if speaking_input and speaking_input.reference_text:
        logger.info("[eval:correction] skipped (reference_text provided) answer_id=%s turn=%s", answer_id, turn_order)
        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "correction_skipped": True,
                "correction_reason": "reference_text provided",
            },
        }

    transcribed_text = speaking_input.transcribed_text if speaking_input else None

    if not transcribed_text:
        # Nothing to correct (e.g. no speech recognized at all). This node now runs
        # BEFORE strict_validity_check (see graphConfig.build_graph), so returning
        # status="error" here would abort the whole turn -- and with it, the entire
        # multi-turn answer -- before validity_node's own "audio.no_speech" rule ever
        # gets a chance to classify this turn as reject_or_zero. Skip gracefully instead.
        logger.info("[eval:correction] skipped (empty transcript) answer_id=%s turn=%s", answer_id, turn_order)
        return {
            **state,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "correction_skipped": True,
                "correction_reason": "empty transcript",
            },
        }

    logger.info("[eval:correction] correcting transcript answer_id=%s turn=%s", answer_id, turn_order)

    try:
        corrected_transcript = correct_transcript(transcribed_text)

        if speaking_input:
            speaking_input.corrected_transcript = corrected_transcript

        logger.info("[eval:correction] done answer_id=%s turn=%s changed=%s", answer_id, turn_order, corrected_transcript != transcribed_text)

        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "original_transcript": transcribed_text,
                "correction_applied": corrected_transcript != transcribed_text,
            },
        }

    except Exception as exc:
        # Same reasoning as the empty-transcript branch above: correction now runs
        # before validity for every turn, so a transient OpenAI failure here must not
        # abort the whole answer -- fall back to the raw transcript (corrected_transcript
        # left unset) and let validity/pronunciation continue on that instead.
        logger.exception("[eval:correction] failed, falling back to raw transcript answer_id=%s turn=%s", answer_id, turn_order)
        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "correction_error": str(exc),
            },
        }
