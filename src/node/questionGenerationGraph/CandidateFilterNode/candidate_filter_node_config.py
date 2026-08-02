from concurrent.futures import ThreadPoolExecutor

from node.questionGenerationGraph.constants import (
    ALLOWED_SUB_ATTRIBUTES,
    DUPLICATE_THRESHOLD,
    EMBEDDING_MODEL,
    FILTER_REASON_CODES,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
    TokenCall,
    question_embedding_text,
)
from schemas.question_generation import PracticeQuestionCandidate


def rule_violations(
    candidate: PracticeQuestionCandidate,
) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    text = candidate.prompt_text.strip()
    words = text.split()
    ascii_letters = sum(
        character.isascii() and character.isalpha()
        for character in text
    )
    letters = sum(character.isalpha() for character in text)
    if len(words) < 6 or len(words) > 80:
        violations.append(
            (
                "LENGTH_OUT_OF_RANGE",
                f"prompt has {len(words)} words; expected 6..80",
            )
        )
    if letters == 0 or ascii_letters / letters < 0.9:
        violations.append(
            (
                "NOT_ENGLISH",
                "fewer than 90% of alphabetic characters are ASCII English",
            )
        )
    allowed = ALLOWED_SUB_ATTRIBUTES.get(candidate.target_construct)
    if allowed is None:
        violations.append(
            (
                "CRITERION_UNKNOWN",
                f"{candidate.target_construct} is not a framework criterion",
            )
        )
    elif candidate.target_sub_attribute not in allowed:
        rendered = (
            "null"
            if candidate.target_sub_attribute is None
            else candidate.target_sub_attribute
        )
        violations.append(
            (
                "SUB_ATTRIBUTE_NOT_ALLOWED",
                f"{rendered} is not allowed for {candidate.target_construct}",
            )
        )
    assert all(reason in FILTER_REASON_CODES for reason, _ in violations)
    return violations


def candidate_filter_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    survivors = []
    rejected = list(state.get("rejected", []))
    reasons: set[str] = set()
    embeddings = {}
    cosines = []
    topic_name = state["topic"][0]
    tokens = state["token_calls"]

    # Lọc bằng luật (thuần CPU) trước, rồi mới nhúng -- ứng viên trượt luật
    # không tốn call embedding nào.
    passed_rules = []
    for candidate in state["candidates"]:
        violations = rule_violations(candidate)
        if violations:
            reason, detail = violations[0]
            reasons.update(item[0] for item in violations)
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": reason,
                    "detail": detail,
                    "candidate": candidate.model_dump(),
                }
            )
            continue
        passed_rules.append(candidate)

    # Mỗi ứng viên 1 lượt embed + 1 lượt tra Chroma, hoàn toàn độc lập nhau ->
    # chạy song song, tổng thời gian bằng lượt chậm nhất thay vì cộng dồn.
    def _embed_and_score(candidate):
        embedding, token_count = runtime.embed(
            question_embedding_text(topic_name, candidate.prompt_text)
        )
        return candidate, embedding, token_count, runtime.max_similarity(embedding)

    if passed_rules:
        with ThreadPoolExecutor(max_workers=len(passed_rules)) as pool:
            scored = list(pool.map(_embed_and_score, passed_rules))
    else:
        scored = []

    for candidate, embedding, token_count, similarity in scored:
        tokens.append(
            TokenCall(
                role="embedding",
                mode="question-filter",
                model=EMBEDDING_MODEL,
                input=token_count,
                output=0,
                reasoning=0,
                cached_input=0,
                response_id="",
            )
        )
        cosines.append(similarity)
        if similarity >= DUPLICATE_THRESHOLD:
            reasons.add("DUPLICATE_COSINE")
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": "DUPLICATE_COSINE",
                    "detail": (
                        f"cosine {similarity:.6f} >= "
                        f"{DUPLICATE_THRESHOLD:.2f}"
                    ),
                    "candidate": candidate.model_dump(),
                }
            )
            continue
        survivors.append(candidate)
        embeddings[candidate.candidate_id] = embedding
    return {
        "survivors": survivors,
        "rejected": rejected,
        "filter_reasons": reasons,
        "survivor_embeddings": embeddings,
        "cosines": cosines,
    }
