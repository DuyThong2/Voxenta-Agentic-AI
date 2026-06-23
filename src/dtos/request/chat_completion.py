from typing import List, Literal, Optional

from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat-completions request.

    Deliberately a plain BaseModel, not _CamelMessage: this is the wire
    format Tavus's Custom LLM integration sends, and OpenAI's own API is
    snake_case, so this is the one DTO pair in the service that must stay
    snake_case rather than follow the camelCase convention used elsewhere.
    """

    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
