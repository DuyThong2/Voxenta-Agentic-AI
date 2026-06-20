import os

from node.evalGraph.GraphState import GraphState
from schemas.enums import SpeakingMode
from utils.speech_client import normalize_text, transcribe


def start_node(state: GraphState) -> dict:
    speaking_input = state.get("speaking_input")

    if speaking_input is None:
        return {
            **state,
            "status": "error",
            "error": "speaking_input is required for start_node",
        }

    audio_path = speaking_input.audio_path
    if not audio_path:
        return {
            **state,
            "status": "error",
            "error": "speaking_input.audio_path is required",
        }

    if not os.path.exists(audio_path):
        return {
            **state,
            "status": "error",
            "error": f"Audio file not found: {audio_path}",
        }

    try:
        transcript = transcribe(audio_path, speaking_input.language)

        if speaking_input.mode == SpeakingMode.UNSCRIPTED and not transcript:
            return {
                **state,
                "status": "error",
                "error": "Audio transcription failed for unscripted mode",
                "metadata": {
                    **state.get("metadata", {}),
                    "recognition_reason": "not_recognized",
                },
            }

        normalized_transcript = normalize_text(transcript)

        if speaking_input.mode == SpeakingMode.SCRIPTED:
            # For scripted mode, reference_text is authoritative and should be normalized.
            speaking_input.reference_text = normalize_text(speaking_input.reference_text)

        speaking_input.transcribed_text = transcript

        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "transcription_text": transcript,
            },
        }

    except Exception as exc:
        return {
            **state,
            "status": "error",
            "error": str(exc),
        }
