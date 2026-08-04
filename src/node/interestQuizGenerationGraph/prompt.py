from node.interestQuizGenerationGraph.constants import MAX_STATEMENT_WORDS
from schemas.interest_quiz_generation import InterestQuizItemGenerationRequest

# Ví dụ lấy nguyên văn từ practice/interest-quiz-seed.json (bộ tĩnh gốc đang dùng) -- giữ đúng
# văn phong/độ dài/6 nhãn dimension hệ thống đã chốt, không dùng item O*NET tiếng Anh vì khác
# taxonomy (xem task/implement/13-..., mục 1: few-shot phải khớp construct thật đang dùng).
_FEW_SHOT_EXAMPLES = [
    {
        "dimension_per_statement": ["ENTERTAINMENT_MEDIA", "TECH_GAMING", "FUTURE_SCIENCE"],
        "statements": [
            "Sắp ảnh thành áp phích cho buổi diễn",
            "Phác thảo màn chơi vượt chướng ngại",
            "Khám phá vì sao xà phòng rửa sạch dầu",
        ],
    },
    {
        "dimension_per_statement": ["SPORTS_HEALTH", "PEOPLE_SOCIETY", "TRAVEL_PLACES"],
        "statements": [
            "Chọn bài tập giãn cơ sau giờ học",
            "Tổ chức trò làm quen cho nhóm mới",
            "Tìm lối đi ngắn nhất trong sơ đồ vườn thú",
        ],
    },
    {
        "dimension_per_statement": ["TECH_GAMING", "PEOPLE_SOCIETY", "FUTURE_SCIENCE"],
        "statements": [
            "Vẽ sơ đồ nút bấm cho máy bán nước",
            "Tìm hiểu vì sao có nhóm trưởng tự nhiên",
            "Xem cách bóng giấy rơi chậm hơn viên đá",
        ],
    },
]


def dimension_triplets(dimensions: list[str], count: int) -> list[list[str]]:
    """Phân công trước bộ ba chiều cho từng item, cân bằng số lần mỗi chiều xuất hiện.

    Vì sao phải phân công thay vì để model tự chọn: sinh song song thì mỗi lượt KHÔNG BIẾT
    các lượt kia viết gì, nên nếu để tự do sẽ ra hai item cùng nói về game trong khi một
    chiều không câu nào chạm tới. Bản một-lượt trước đây không gặp chuyện này vì model nhìn
    thấy toàn bộ 7 item cùng lúc -- đó cũng chính là thứ khiến nó mất 47 giây.

    Tham lam theo "chiều ít dùng nhất trước", kèm tránh lặp lại nguyên một tổ hợp đã dùng.
    """
    usage = {code: 0 for code in dimensions}
    used_sets: set[frozenset] = set()
    triplets: list[list[str]] = []
    for _ in range(count):
        ordered = sorted(dimensions, key=lambda code: (usage[code], dimensions.index(code)))
        chosen = ordered[:3]
        if frozenset(chosen) in used_sets and len(dimensions) > 3:
            # Đổi phần tử thứ ba sang chiều ít dùng kế tiếp để không lặp nguyên tổ hợp.
            for alternative in ordered[3:]:
                if frozenset(chosen[:2] + [alternative]) not in used_sets:
                    chosen = chosen[:2] + [alternative]
                    break
        used_sets.add(frozenset(chosen))
        for code in chosen:
            usage[code] += 1
        triplets.append(chosen)
    return triplets


# Bối cảnh phân công cho từng lượt sinh. Cần vì các lượt chạy SONG SONG nên không thấy nhau:
# đo thật lần đầu ra hai item cùng "làm playlist nhạc" và hai item cùng "tùy chỉnh nhân vật
# game", rồi bộ lọc trùng cắt mất một item (6/7). Bản một-lượt trước đây không gặp chuyện này
# vì model nhìn thấy hết các item cùng lúc -- nhưng đó cũng là thứ khiến nó chậm gấp bốn.
#
# Phân công bối cảnh rẻ hơn nhiều so với cho các lượt "nói chuyện" với nhau, và vẫn giữ được
# đúng thứ cần: các item không đè ý nhau.
_ITEM_CONTEXTS = [
    "ở trường, giờ ra chơi hoặc sinh hoạt lớp",
    "ở nhà, lúc rảnh sau giờ học",
    "đi chơi cuối tuần với bạn bè",
    "trên mạng: điện thoại, máy tính, mạng xã hội",
    "trong câu lạc bộ hoặc hoạt động ngoại khoá",
    "khi ở ngoài trời hoặc di chuyển trong thành phố",
    "khi làm việc nhóm cho một dự án hoặc bài tập",
]


def build_single_item_prompt(
    request: InterestQuizItemGenerationRequest,
    assigned_dimensions: list[str],
    context_index: int = 0,
) -> str:
    """Prompt cho MỘT item với bộ ba chiều đã được phân công sẵn.

    Khác bản batch ở hai chỗ, và cả hai đều là lý do nó nhanh: model không phải tự cân bằng
    chiều (đã phân công), và không phải trải 14 ứng viên rồi chọn 7 -- chỉ cân nhắc vài
    phương án cho đúng một item.
    """
    base = build_interest_quiz_prompt(request)
    head, _, _ = base.partition("Diversity instruction")
    context = _ITEM_CONTEXTS[context_index % len(_ITEM_CONTEXTS)]
    return (
        head
        + f"""Write EXACTLY ONE triplet, using these 3 dimensions in this order:
{assigned_dimensions[0]}, {assigned_dimensions[1]}, {assigned_dimensions[2]}.

Set all 3 activities in this context: {context}.
Other items are being written for other contexts, so stay inside yours -- do not write about
making playlists, customising game characters, or any other theme unless it genuinely fits
this context.

Consider a few candidate triplets internally, then return the single best one.

Return structured data only."""
    )


def build_interest_quiz_prompt(request: InterestQuizItemGenerationRequest) -> str:
    # Danh mục chiều đến từ bảng interest_dimension bên vox (Java gửi kèm request), không
    # gắn cứng ở đây -- admin thêm chiều là prompt tự có ngay.
    dimensions_block = ", ".join(request.effective_dimensions())
    examples_block = "\n".join(
        f"- dimensions={example['dimension_per_statement']}, "
        f"statements={example['statements']}"
        for example in _FEW_SHOT_EXAMPLES
    )
    existing_block = (
        "\n".join(f"- {statement}" for statement in request.existing_statements)
        or "(chưa có)"
    )

    return f"""You are an expert psychometrician writing forced-choice interest-inventory
items for Vietnamese high-school students, inside a speaking-practice app.

Each item is a triplet of 3 short everyday activities, one per dimension below, used to infer
which topics a student will find engaging (NOT a career/ability test -- keep activities
familiar, low-stakes, age-appropriate, in Vietnamese).

Dimensions (each triplet must use 3 DIFFERENT ones): {dimensions_block}.

Format rules:
1. Each statement is a short activity, under {MAX_STATEMENT_WORDS} words, in Vietnamese.
2. All 3 statements in one triplet must feel equally ordinary/likeable -- no statement should
   sound obviously "better" or more virtuous than the other two (social-desirability
   matching). Explain this balance briefly in desirability_check.
3. Do NOT repeat or rephrase any statement already listed below.
4. Generate up to {request.max_items} triplets in this one response.
5. CRITICAL -- each statement must be an activity the student can picture themselves CHOOSING
   TO DO, more than once, because they enjoy it. It must NOT be a one-off micro-observation,
   a physics puzzle, or a riddle. The student answers "which is most/least like me", so a
   statement they cannot see themselves in is useless.
   BAD (do not produce anything like these -- they are curiosities, not interests):
     - "Xem vì sao bánh quy mềm đi khi để hở"
     - "So sánh vì sao ly kim loại lạnh lâu hơn"
     - "Thử đo bóng đổ để đoán giờ ngoài sân"
     - "Đoán lý do một tin nhắn dễ bị hiểu sai"
   Avoid openers of the form "Xem vì sao...", "So sánh vì sao...", "Đoán lý do..." entirely.
   GOOD shape: a concrete thing the student does or makes with/for other people, a hobby, or a
   recurring task they would pick up willingly -- e.g. "Dựng video ngắn cho câu lạc bộ",
   "Hướng dẫn bạn mới cách chơi", "Lên lịch tập cho cả nhóm".

Examples (style/length/dimension-set to match, do NOT copy verbatim):
{examples_block}

Statements already in the bank (do not repeat/rephrase these):
{existing_block}

Diversity instruction (Verbalized Sampling): first internally consider several plausible
candidate triplets, then verbalize a spread of {request.max_items * 2} candidates across
different dimension combinations and everyday-activity themes with your estimated probability
of each being a good, balanced triplet, and finally return the best {request.max_items}
selected from that spread -- do not just return the single most typical/obvious set.

Return structured data only."""
