from practice_generation.models import (
    DifficultyFeatures,
    EvaluationGuide,
    PracticeQuestionCandidate,
    difficulty_rank,
)
from practice_generation.question_pipeline import (
    ALLOWED_SUB_ATTRIBUTES,
    FILTER_REASON_CODES,
    HARD_CAP,
    normalize_name,
    percentile,
    question_embedding_text,
    rule_violations,
)


def test_difficulty_rank_is_derived_and_clamped() -> None:
    assert difficulty_rank(
        DifficultyFeatures(
            here_and_now=True,
            num_elements=1,
            reasoning_type="description",
            abstractness="concrete_personal",
        )
    ) == 1
    assert difficulty_rank(
        DifficultyFeatures(
            here_and_now=False,
            num_elements=5,
            reasoning_type="hypothetical",
            abstractness="abstract",
        )
    ) == 6


def test_generation_invariants() -> None:
    assert HARD_CAP == 50
    assert normalize_name("  Chủ đề   Mới ") == "chu de moi"
    assert question_embedding_text("Games", "What do you play?").startswith(
        "Topic: Games"
    )
    assert percentile([0.1, 0.2, 0.8, 0.9], 0.95) == 0.9


def test_filter_codes_and_sub_attribute_taxonomy_are_closed() -> None:
    assert FILTER_REASON_CODES == {
        "NOT_ENGLISH",
        "LENGTH_OUT_OF_RANGE",
        "MISSING_FIELD",
        "SUB_ATTRIBUTE_NOT_ALLOWED",
        "CRITERION_UNKNOWN",
        "DUPLICATE_COSINE",
    }
    assert ALLOWED_SUB_ATTRIBUTES["PRONUNCIATION"] == {None}
    assert ALLOWED_SUB_ATTRIBUTES["FLUENCY"] == {None}
    assert len(
        set().union(
            ALLOWED_SUB_ATTRIBUTES["GRAMMAR"],
            ALLOWED_SUB_ATTRIBUTES["VOCABULARY"],
            ALLOWED_SUB_ATTRIBUTES["COHERENCE"],
        )
    ) == 13


def test_imperative_prompt_is_valid_but_invalid_sub_attribute_is_not() -> None:
    guide = EvaluationGuide(
        expected_content="A description.",
        key_points="Place and reason.",
        acceptable_responses="Any relevant personal answer.",
        off_topic_examples="An unrelated story.",
        scoring_hints="Use the target construct.",
        common_mistakes="Missing supporting detail.",
    )
    valid = PracticeQuestionCandidate(
        candidate_id="valid-1",
        difficulty_features=DifficultyFeatures(
            here_and_now=True,
            num_elements=2,
            reasoning_type="description",
            abstractness="concrete_personal",
        ),
        target_construct="PRONUNCIATION",
        target_sub_attribute=None,
        vstep_part=1,
        prompt_text="Describe a place you often visit after school.",
        suggested_ideas=["where it is", "why you go"],
        planning_time_seconds=20,
        max_response_seconds=60,
        max_followup_seconds=0,
        evaluation_guide=guide,
    )
    assert rule_violations(valid) == []

    invalid = valid.model_copy(update={"target_sub_attribute": "word_stress"})
    assert rule_violations(invalid)[0][0] == "SUB_ATTRIBUTE_NOT_ALLOWED"
