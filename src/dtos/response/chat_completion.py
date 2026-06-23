from typing import List, Literal, Optional

from pydantic import BaseModel

from dtos.request.chat_completion import ChatMessage, ToolCall


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming chat-completions response. Snake_case
    on purpose — see dtos.request.chat_completion.ChatCompletionRequest."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: Optional[str] = None
    choices: List[ChatCompletionChoice]


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop"]] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible SSE chunk shape for stream=true."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: Optional[str] = None
    choices: List[ChatCompletionChunkChoice]
