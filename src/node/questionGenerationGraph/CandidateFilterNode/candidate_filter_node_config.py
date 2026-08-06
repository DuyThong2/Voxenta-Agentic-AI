from concurrent.futures import ThreadPoolExecutor

from node.questionGenerationGraph.constants import (
    ALLOWED_SUB_ATTRIBUTES,
    ALLOWED_TENSES,
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
    requested_sub_attribute: str | None = None,
    requested_tense: str | None = None,
) -> list[tuple[str, str]]:
    """Vi pham cung cua mot ung vien.

    `requested_sub_attribute` la sub-attribute Java YEU CAU (None = khong chi dinh).
    None -> chap nhan ca ung vien null lan ung vien co gia tri hop le trong taxonomy.
    Co gia tri -> ung vien phai dung y gia tri do.

    `requested_tense` cung nguyen tac. O day CHI kiem phan CUNG -- nhan tra ve co nam trong
    tap dong va co khop cai Java xin khong. Viec "moc thoi gian cua cau co THAT SU ep duoc
    thi do khong" thi khong kiem duoc bang luat: bat tu khoa (did/last/will/would) chi la do
    dau hieu be mat, va cau "Tell me about a time your school changed" ep qua khu ma khong
    chua tu nao trong danh sach. Phan do giao cho evaluator -- xem evaluator_prompt.
    """
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
    rendered = (
        "null"
        if candidate.target_sub_attribute is None
        else candidate.target_sub_attribute
    )
    if allowed is None:
        violations.append(
            (
                "CRITERION_UNKNOWN",
                f"{candidate.target_construct} is not a framework criterion",
            )
        )
    elif requested_sub_attribute is not None:
        # Java co chi dinh sub-attribute cu the -> phai dung cai do, khong duoc lech.
        if candidate.target_sub_attribute != requested_sub_attribute:
            violations.append(
                (
                    "SUB_ATTRIBUTE_NOT_ALLOWED",
                    f"{rendered} does not match requested "
                    f"{requested_sub_attribute}",
                )
            )
    elif (
        candidate.target_sub_attribute is not None
        and candidate.target_sub_attribute not in allowed
    ):
        # Java KHONG chi dinh (chua co du lieu diem yeu) -> null la hop le, nghia la "luyen
        # tieu chi nay noi chung". Truoc day null bi loai cho GRAMMAR/VOCABULARY/COHERENCE,
        # khien hoc sinh moi khong bao gio sinh duoc cau nao -- be tac: khong luyen duoc thi
        # khong co diem yeu, khong co diem yeu thi mai gui null.
        violations.append(
            (
                "SUB_ATTRIBUTE_NOT_ALLOWED",
                f"{rendered} is not allowed for {candidate.target_construct}",
            )
        )
    if candidate.target_tense not in ALLOWED_TENSES:
        violations.append(
            (
                "TENSE_MISMATCH",
                f"{candidate.target_tense} is not in the closed tense taxonomy",
            )
        )
    elif requested_tense is not None and candidate.target_tense != requested_tense:
        violations.append(
            (
                "TENSE_MISMATCH",
                f"{candidate.target_tense} does not match requested {requested_tense}",
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
        violations = rule_violations(
            candidate, state["criterion"][1], state.get("target_tense")
        )
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
        return (
            candidate,
            embedding,
            token_count,
            runtime.max_similarity(embedding, state.get("exclude_question_ids")),
        )

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
        # KHONG con loai ung vien vi trung voi LICH SU kho nua (truoc: cosine >= 0,92 ->
        # DUPLICATE_COSINE). Van tinh va giu `cosines` de theo doi, chi bo phan CHAN.
        #
        # Vi sao bo: cong nay khoa cung chinh cai no phuc vu. Trong MOT chu de, khi hoc sinh da
        # luyen het cau co san thi cau moi TAT NHIEN se giong cau cu -- cung chu de, cung tieu
        # chi, cung bac thi khong con bao nhieu cach hoi khac nhau. Chan lai nghia la:
        #
        #     doc kho: rong  ->  nho LLM soan  ->  soan ra cai giong cau cu  ->  vut
        #     ->  survivors = []  ->  pool_exhausted, VINH VIEN
        #
        # Thu chan no lai la thu hoc sinh khong duoc phep dung nua. Do tren du lieu that
        # 2026-08-05: 64 vector trong Chroma nhung chi 13 cau trong Postgres -- 51 vector mo coi
        # tu lan xoa DB truoc, va cong nay dang so ban nhap moi voi ca nhung cau KHONG CON TON
        # TAI. Khong co duong nao go duoc bang danh sach loai tru, vi chung khong nam trong
        # bat ky bang nao cua Postgres.
        #
        # Danh doi da chap nhan (quyet dinh cua nguoi dung): kho SE co cau na na nhau. Doi lai,
        # lop chan LAP trong MOT BUOI van con nguyen -- xem maxSimilarities o
        # PracticeQuestionSelectionService.pickOne, nguong 0,85, so voi cau da phat trong chinh
        # phien do. Do la lop hoc sinh cam nhan duoc; trung voi cau cua thang truoc thi khong.
        cosines.append(similarity)
        survivors.append(candidate)
        embeddings[candidate.candidate_id] = embedding
    return {
        "survivors": survivors,
        "rejected": rejected,
        "filter_reasons": reasons,
        "survivor_embeddings": embeddings,
        "cosines": cosines,
    }
