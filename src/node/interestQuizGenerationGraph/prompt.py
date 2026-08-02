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
