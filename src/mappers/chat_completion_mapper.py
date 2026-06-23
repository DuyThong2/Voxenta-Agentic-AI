"""Translates between the OpenAI chat-completions wire shape (Tavus's Custom
LLM integration) and the FollowUpGraphState used by node/followUpDecisionGraph.
"""

import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from dtos.request.chat_completion import ChatMessage
from dtos.response.chat_completion import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionResponse,
)
from node.state_models import QuestionContext
from utils.text_utils import word_count

CLOSING_REPLY = "Thank you, that's all for this question."

_QUESTION_CONTEXT_MARKER = re.compile(r"<question_context>(.*?)</question_context>", re.DOTALL)


def _extract_question_context(messages: List[ChatMessage]) -> Optional[QuestionContext]:
    """Pull QuestionContext out of a <question_context>{snake_case JSON}</question_context>
    marker in a system message. Missing marker or invalid JSON -> None: this
    is supporting context for the LLM decision, not required data."""
    for message in messages:
        if message.role != "system":
            continue
        match = _QUESTION_CONTEXT_MARKER.search(message.content)
        if not match:
            continue
        try:
            return QuestionContext.model_validate_json(match.group(1).strip())
        except (ValueError, ValidationError):
            return None
    return None


def _merge_consecutive_same_role(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
    """Normalize the raw transcript into ordered (role, content) blocks: drop
    system messages and merge consecutive same-role messages (joined with
    "\\n") into one block, so turn-pairing doesn't depend on strict
    user/assistant alternation in the raw message list."""
    blocks: List[Tuple[str, str]] = []
    for message in messages:
        if message.role == "system":
            continue
        if blocks and blocks[-1][0] == message.role:
            role, content = blocks[-1]
            blocks[-1] = (role, f"{content}\n{message.content}")
        else:
            blocks.append((message.role, message.content))
    return blocks


def build_followup_state_from_messages(messages: List[ChatMessage]) -> Dict[str, Any]:
    question = _extract_question_context(messages)

    turns: List[Dict[str, Any]] = []
    pending_prompt: Optional[str] = None
    for role, content in _merge_consecutive_same_role(messages):
        if role == "assistant":
            pending_prompt = content
            continue

        turn_order = len(turns) + 1
        turns.append({
            "turn_order": turn_order,
            "turn_type": "MAIN" if turn_order == 1 else "FOLLOWUP",
            "prompt_text": pending_prompt,
            "transcript": content,
            "word_count": word_count(content),
        })
        pending_prompt = None

    if not turns:
        raise ValueError("messages must contain at least one user message")

    current_turn = turns[-1]
    previous_turns = turns[:-1]

    return {
        "question": question,
        "turn_order": current_turn["turn_order"],
        "prompt_text": current_turn["prompt_text"],
        "current_turn": current_turn,
        "turns": previous_turns,
        "status": "idle",
    }


def resolve_reply_content(decision: Dict[str, Any]) -> str:
    next_prompt_text = decision.get("next_prompt_text")
    if decision.get("should_continue") and str(next_prompt_text or "").strip():
        return next_prompt_text
    return CLOSING_REPLY


def build_chat_completion_response(model: Optional[str], reply_content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(message=ChatMessage(role="assistant", content=reply_content)),
        ],
    )


def build_chat_completion_chunk(
    chunk_id: str,
    created: int,
    model: Optional[str],
    *,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ChatCompletionChunk:
    delta = ChatCompletionChunkDelta(role="assistant" if content is not None else None, content=content)
    return ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)],
    )
