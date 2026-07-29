from practice_generation.question_selection import _cosine


def test_cosine_distinguishes_same_and_orthogonal_vectors() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
