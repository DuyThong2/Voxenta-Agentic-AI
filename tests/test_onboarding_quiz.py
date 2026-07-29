from practice_generation.onboarding_quiz import (
    CandidateGroup,
    DesirabilityCheck,
    QuizGenerationBatch,
    TripletCandidate,
    select_items,
)


def _candidate(index: int, probability: float = 0.2) -> TripletCandidate:
    return TripletCandidate(
        probability=probability,
        dimension_per_statement=[
            "ENTERTAINMENT_MEDIA",
            "TECH_GAMING",
            "SPORTS_HEALTH",
        ],
        statements=[
            f"Xem một bộ phim mới số {index}",
            f"Thử một trò chơi mới số {index}",
            f"Tập một môn thể thao số {index}",
        ],
        desirability_check=DesirabilityCheck(
            balanced=True,
            note="Ba hoạt động đều cụ thể và trung tính.",
        ),
    )


def test_select_items_is_seeded_and_unique() -> None:
    batch = QuizGenerationBatch(
        groups=[
            CandidateGroup(
                candidates=[_candidate(group * 10 + index) for index in range(5)]
            )
            for group in range(3)
        ]
    )

    first = select_items(batch, count=3, seed=17)
    second = select_items(batch, count=3, seed=17)

    assert [item.statements for item in first] == [
        item.statements for item in second
    ]
    assert len({tuple(item.statements) for item in first}) == 3
