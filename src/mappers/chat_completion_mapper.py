"""Translates between the OpenAI chat-completions wire shape (Tavus's Custom
LLM integration) and the FollowUpGraphState used by node/followUpDecisionGraph.
"""

import re
import time
import uuid
from json import JSONDecodeError, loads
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from dtos.request.chat_completion import ChatMessage, ToolCall, ToolCallFunction
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
_ANSWER_ID_MARKER = re.compile(r"<answer_id>(.*?)</answer_id>", re.DOTALL)
_QUESTION_TEXT_MARKER = re.compile(r"<question_text>(.*?)</question_text>", re.DOTALL)
_INSTRUCTION_MARKER = re.compile(r"<instruction>(.*?)</instruction>", re.DOTALL)


def _extract_tag_content(content: str, pattern: re.Pattern[str]) -> Optional[str]:
    match = pattern.search(content)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _repair_question_context_json(raw_value: str) -> str:
    """Tavus sometimes relays our system prompt back with a broken
    `evaluation_guide` field (`"evaluation_guide":}`) even though WPF sent
    valid JSON originally. Strip only that malformed fragment so the rest of
    the question context remains usable."""
    repaired = re.sub(r',\s*"evaluation_guide"\s*:\s*}', "}", raw_value)
    repaired = re.sub(r'"evaluation_guide"\s*:\s*}', '"evaluation_guide": null}', repaired)
    return repaired


def _fallback_question_context_from_marked_message(content: str) -> Optional[QuestionContext]:
    """Best-effort recovery when Tavus preserves the surrounding markers but
    mangles the JSON inside <question_context>. Keep only fields we can parse
    with high confidence from the broken payload and surrounding tags."""
    question_match = _QUESTION_CONTEXT_MARKER.search(content)
    if not question_match:
        return None

    raw_value = question_match.group(1).strip()
    repaired_value = _repair_question_context_json(raw_value)

    try:
        parsed = loads(repaired_value)
    except JSONDecodeError:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    if "question_text" not in parsed:
        parsed["question_text"] = _extract_tag_content(content, _QUESTION_TEXT_MARKER)

    if not parsed.get("question_text"):
        instruction_text = _extract_tag_content(content, _INSTRUCTION_MARKER)
        if instruction_text:
            parsed["instruction_text"] = instruction_text

    try:
        question = QuestionContext.model_validate(parsed)
    except ValidationError:
        return None

    if not any(question.model_dump(exclude_none=True).values()):
        return None
    return question


def _last_marked_system_message(messages: List[ChatMessage]) -> Optional[Tuple[int, str]]:
    """Find the most recent system message that actually carries our own
    <question_context>/<answer_id> markers — scanning backwards but skipping any system
    message that doesn't match, not just blindly trusting whichever system message is
    physically last. One Tavus conversation now spans the whole exam, and WPF marks each
    question boundary with conversation.overwrite-context (always lands as a system
    message), but nothing guarantees Tavus's own pipeline never injects an unrelated system
    message (e.g. around a tool call) after that — if we trusted "the last system message"
    unconditionally, that unrelated message would be mistaken for the question boundary:
    question_context/answer_id would wrongly resolve to None, and turn-counting would wrongly
    start after that unrelated message instead of after our real marker, silently dropping
    this question's earlier turns from the signals the decision is based on."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "system":
            continue
        if _QUESTION_CONTEXT_MARKER.search(message.content) or _ANSWER_ID_MARKER.search(message.content):
            return index, message.content
    return None


def _extract_question_context(messages: List[ChatMessage]) -> Optional[QuestionContext]:
    """Pull QuestionContext out of a <question_context>{snake_case JSON}</question_context>
    marker in the most recent *marked* system message. Missing marker or invalid JSON ->
    None: this is supporting context for the LLM decision, not required data."""
    found = _last_marked_system_message(messages)
    if found is None:
        return None
    _, content = found
    match = _QUESTION_CONTEXT_MARKER.search(content)
    if not match:
        return None
    try:
        return QuestionContext.model_validate_json(match.group(1).strip())
    except (ValueError, ValidationError):
        return _fallback_question_context_from_marked_message(content)


def extract_answer_id(messages: List[ChatMessage]) -> Optional[str]:
    """Pull the answer_id WPF embeds via <answer_id>{guid}</answer_id> next to
    the question_context marker in the most recent *marked* system message, so
    /v1/chat/completions can correlate this conversation with the turns
    /turns/archive has been persisting. Not prefixed with "_" like
    _extract_question_context: this one is called from controller/tavus_controller.py,
    a different module."""
    found = _last_marked_system_message(messages)
    if found is None:
        return None
    _, content = found
    match = _ANSWER_ID_MARKER.search(content)
    return match.group(1).strip() if match else None


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
    """One Tavus conversation now spans the whole exam, so `messages` can contain turns
    from previous questions too. Scope turn-counting to whatever comes after the most
    recent *marked* system message — the boundary WPF draws each time it calls
    OverwriteContextAsync for a new question — so turn_order/turns/word counts never
    bleed across questions (or get cut short by an unrelated system message — see
    _last_marked_system_message)."""
    question = _extract_question_context(messages)
    found = _last_marked_system_message(messages)
    scoped_messages = messages if found is None else messages[found[0] + 1:]

    turns: List[Dict[str, Any]] = []
    pending_prompt: Optional[str] = None
    for role, content in _merge_consecutive_same_role(scoped_messages):
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


def build_tavus_message_debug_snapshot(messages: List[ChatMessage]) -> Dict[str, Any]:
    """Produce a compact, log-friendly view of the Tavus payload so we can
    verify which system marker was chosen, how the current question was scoped,
    and which turns were reconstructed for the follow-up decision."""
    found = _last_marked_system_message(messages)
    question = _extract_question_context(messages)
    answer_id = extract_answer_id(messages)
    scoped_messages = messages if found is None else messages[found[0] + 1:]

    merged_blocks = [
        {"role": role, "content": content}
        for role, content in _merge_consecutive_same_role(scoped_messages)
    ]

    snapshot: Dict[str, Any] = {
        "message_count": len(messages),
        "roles": [message.role for message in messages],
        "selected_marker_index": found[0] if found is not None else None,
        "selected_marker_preview": (
            found[1][:500] if found is not None else None
        ),
        "answer_id": answer_id,
        "question_context": question.model_dump(mode="json") if question is not None else None,
        "scoped_message_count": len(scoped_messages),
        "scoped_messages": [message.model_dump(mode="json") for message in scoped_messages],
        "merged_blocks": merged_blocks,
    }

    try:
        state = build_followup_state_from_messages(messages)
        snapshot["followup_state"] = state
    except ValueError as exc:
        snapshot["followup_state_error"] = str(exc)

    return snapshot


def resolve_reply_content(decision: Dict[str, Any]) -> str:
    next_prompt_text = decision.get("next_prompt_text")
    if decision.get("should_continue") and str(next_prompt_text or "").strip():
        return next_prompt_text
    return CLOSING_REPLY


def build_end_question_tool_call() -> ToolCall:
    """Tavus relays tool_calls to the WPF client as an app-message instead of
    executing them — this is the "question done" signal WPF listens for,
    replacing the old approach of string-matching the spoken closing reply."""
    return ToolCall(id=f"call_{uuid.uuid4().hex}", function=ToolCallFunction(name="end_question"))


def build_chat_completion_response(
    model: Optional[str],
    reply_content: str,
    tool_calls: Optional[List[ToolCall]] = None,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=reply_content, tool_calls=tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            ),
        ],
    )


def build_chat_completion_chunk(
    chunk_id: str,
    created: int,
    model: Optional[str],
    *,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
) -> ChatCompletionChunk:
    delta = ChatCompletionChunkDelta(
        role="assistant" if (content is not None or tool_calls is not None) else None,
        content=content,
        tool_calls=tool_calls,
    )
    return ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)],
    )
