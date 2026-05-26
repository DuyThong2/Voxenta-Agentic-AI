"""LangSmith configuration for observability and debugging."""

import os


def setup_langsmith() -> None:
    """
    Setup LangSmith tracing by mapping environment variables.
    
    Reads from:
    - LANGSMITH_TRACING: Enable/disable tracing
    - LANGSMITH_API_KEY: Your LangSmith API key
    - LANGSMITH_PROJECT: Project name for organizing traces
    - LANGSMITH_ENDPOINT: LangSmith endpoint URL
    
    Sets LangChain environment variables for auto-detection.
    """
    
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "agents-pronunciation")
    langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    # Set LangChain environment variables for LangChain to auto-detect
    if langsmith_tracing and langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint


def get_langsmith_status() -> str:
    """Get readable status of LangSmith configuration."""
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    api_key_set = bool(os.getenv("LANGSMITH_API_KEY"))
    project_name = os.getenv("LANGSMITH_PROJECT", "agents-pronunciation")
    
    if not tracing_enabled:
        return "LangSmith tracing is DISABLED"
    
    if not api_key_set:
        return "LangSmith tracing is ENABLED but LANGSMITH_API_KEY is not set"
    
    return f"LangSmith tracing is ENABLED - Project: {project_name}"
