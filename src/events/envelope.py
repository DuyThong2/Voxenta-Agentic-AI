from schemas.common import _CamelMessage


class EventEnvelope(_CamelMessage):
    """Base for every Kafka event payload.

    `event_type` is the discriminator that lets a small number of topics
    carry many kinds of events (Java branches on this field instead of us
    adding a new topic per event kind).
    """

    event_type: str
    schema_version: int = 1
