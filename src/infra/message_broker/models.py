"""Pydantic models for RabbitMQ message payloads.

Property names use snake_case internally.  When serialising to JSON for
RabbitMQ we emit camelCase keys (``by_alias=True``) so that .NET
MassTransit consumers can deserialise without extra mapping.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _CamelMessage(BaseModel):
    """Base with camelCase alias generation."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class PaperIngestionMessage(_CamelMessage):
    """Inbound message published by the .NET service.

    Corresponds to ``PaperIngestionEvent`` on the .NET side.
    """

    paper_id: str
    paper_name: str
    reference_key: str 
    authors: str 
    publisher: str 
    journal_name: str
    volume: str
    pages: str
    doi: str
    publication_month_year: str
    parsed_text: str


class PaperIngestionCompletedMessage(_CamelMessage):
    """Outbound message published back to the .NET service.

    Corresponds to ``PaperIngestionCompletedEvent`` on the .NET side.
    """

    paper_id: str
    is_success: bool
    error_message: Optional[str] = None
