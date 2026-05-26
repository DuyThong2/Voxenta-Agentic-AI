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
        "speaking_input": SpeakingInput(...),
        "transcribed_text": str
    }
    
    Returns state with:
    {
        "speaking_input": SpeakingInput updated with corrected_transcript,
        "corrected_transcript": str,
        "status": "processing"
    }
    """
    
    speaking_input = state.get("speaking_input")
    transcribed_text = state.get("transcribed_text")
    
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

        next_state = {k: v for k, v in state.items() if k not in ("transcribed_text", "formatted_reference_text")}
        next_state.update({
            "speaking_input": speaking_input,
            "corrected_transcript": corrected_transcript,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "original_transcript": transcribed_text,
                "correction_applied": corrected_transcript != transcribed_text,
            },
        })

        return next_state
    
    except Exception as exc:
        return {
            **state,
            "status": "error",
            "error": f"Transcript correction failed: {str(exc)}",
        }
