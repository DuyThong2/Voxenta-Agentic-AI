"""
Data Transfer Objects (DTOs) and models for the chat application.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    # Chat messages managed by LangGraph reducer
    messages: Annotated[List[BaseMessage], add_messages]
    
