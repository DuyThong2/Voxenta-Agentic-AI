from node.topicGenerationGraph.GraphState import TopicGenerationState
from schemas.topic_generation import KeywordEvidence


def keyword_extraction_node(state: TopicGenerationState) -> dict:
    request = state["request"].model_copy(deep=True)
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for item in request.keyword_evidence:
        normalized = item.keyword.casefold().strip()
        if not normalized:
            continue
        counts[normalized] = max(counts.get(normalized, 0), item.session_count)
        labels.setdefault(normalized, item.keyword.strip())
    request.keyword_evidence = [
        KeywordEvidence(keyword=labels[key], session_count=count)
        for key, count in sorted(
            counts.items(),
            key=lambda entry: (-entry[1], entry[0]),
        )
    ]
    return {"request": request}
