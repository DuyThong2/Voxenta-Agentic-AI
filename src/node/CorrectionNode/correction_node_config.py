"""Script correction node using OpenAI to refine Azure transcription."""

from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.CorrectionNode.correction_node_prompt import SYSTEM_PROMPT


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

    if speaking_input and speaking_input.reference_text:
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
        return {
            **state,
            "status": "error",
            "error": "transcribed_text is required for correction_node",
        }
    
    try:
        corrected_transcript = correct_transcript(transcribed_text)
        
        if speaking_input:
            speaking_input.corrected_transcript = corrected_transcript

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
        return {
            **state,
            "status": "error",
            "error": f"Transcript correction failed: {str(exc)}",
        }
