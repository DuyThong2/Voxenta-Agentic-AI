from practice_generation.topic_service import (
    TopicProposal,
    TopicProposalRequest,
    _deduplicate_evidence_clusters,
    enforce_evidence_caps,
)


def _proposal(**overrides) -> TopicProposal:
    values = {
        "name": "Football and fan culture",
        "interest_dimension": "SPORTS_HEALTH",
        "curriculum_group": "OUT_OF_CURRICULUM",
        "confidence": 0.9,
        "reason_text": "You mentioned football in recent sessions.",
        "distinct_from": "It focuses on supporters rather than playing.",
        "evidence_type": "KEYWORD",
        "evidence_keywords": ["football"],
        "grounded_in_keyword": True,
    }
    values.update(overrides)
    return TopicProposal(**values)


def test_keyword_confidence_uses_supporting_session_count() -> None:
    request = TopicProposalRequest(
        student_id="student-1",
        keyword_evidence=[
            {"keyword": "football", "session_count": 1},
            {"keyword": "music", "session_count": 4},
        ],
    )

    proposals = enforce_evidence_caps([_proposal()], request)

    assert proposals[0].confidence == 0.5


def test_at_most_one_ungrounded_proposal_survives() -> None:
    request = TopicProposalRequest(
        student_id="student-1",
        keyword_evidence=[{"keyword": "football", "session_count": 4}],
    )
    ungrounded = {
        "evidence_type": "INTEREST",
        "evidence_keywords": [],
        "grounded_in_keyword": False,
    }

    proposals = enforce_evidence_caps(
        [
            _proposal(name="Sports nutrition", **ungrounded),
            _proposal(name="Outdoor fitness", **ungrounded),
        ],
        request,
    )

    assert len(proposals) == 1
    assert proposals[0].confidence == 0.4


def test_search_requires_high_confidence() -> None:
    request = TopicProposalRequest(
        student_id="student-1",
        keyword_evidence=[{"keyword": "anime", "session_count": 1}],
        search_keyword=True,
        max_proposals=1,
    )

    proposals = enforce_evidence_caps(
        [
            _proposal(
                confidence=0.89,
                evidence_type="SEARCH",
                evidence_keywords=["anime"],
            )
        ],
        request,
    )

    assert proposals == []


def test_related_keyword_proposals_are_grouped_once() -> None:
    proposals = _deduplicate_evidence_clusters(
        [
            _proposal(
                name="Football during major tournaments",
                evidence_keywords=["football", "World Cup", "Messi"],
            ),
            _proposal(
                name="How star players influence supporters",
                evidence_keywords=["Messi", "football"],
            ),
        ]
    )

    assert [proposal.name for proposal in proposals] == [
        "Football during major tournaments"
    ]
