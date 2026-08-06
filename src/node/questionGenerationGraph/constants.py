import os

from config.chroma_config import settings
from schemas.question_generation import (
    DRAFTER_CANDIDATES as SCHEMA_DRAFTER_CANDIDATES,
)

MODEL = os.getenv("PRACTICE_GENERATION_MODEL", "gpt-5.4")
EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL
HARD_CAP = 80
# Re-export tu schemas: dinh nghia phai nam o tang duoi de DraftBatch bam theo duoc ma khong
# tao vong lap import. Xem chu thich tai cho dinh nghia de biet vi sao la 5.
DRAFTER_CANDIDATES = SCHEMA_DRAFTER_CANDIDATES
MAX_EDITOR_ROUNDS = 3
REFINER_BATCH_SIZE = 4
DUPLICATE_THRESHOLD = 0.92

# --- Chế độ FAST (đường online, học sinh đang chờ) ---------------------------
# Đường nghiên cứu (pipeline.py) giữ nguyên đầy đủ để không đổi số liệu đã đo.
# Đường online chỉ cần 1-2 câu dùng được NGAY, nên cắt các bước chỉ phục vụ đo
# đạc/đánh bóng:
#   - bỏ hẳn lượt evaluator "grouped" (chỉ sinh comparison_total/different cho
#     build_summary, không tham gia quyết định nhận/loại câu nào)
#   - chấm cả lô trong 1 lượt thay vì mỗi câu 1 lượt (3 call -> 1 call)
#   - effort thấp thay vì cao
#   - tối đa 1 vòng sửa: đã có 3 ứng viên, câu nào sửa 1 lần chưa đạt thì bỏ,
#     rẻ hơn nhiều so với cố cứu bằng 2 vòng nữa (mỗi vòng 2 call tuần tự)
#   - bỏ refiner (chỉ là copy-editing, không phải cổng chất lượng)
FAST_EDITOR_ROUNDS = 1
# "medium" chứ không "low": chấm là bước quyết định câu nào tới tay học sinh, hạ
# hẳn xuống low làm chất lượng trôi thấy rõ. Phần tiết kiệm thời gian lớn đã đến
# từ việc bỏ lượt grouped + gộp cả lô vào 1 lượt + chạy song song, không cần vắt
# thêm ở đây.
FAST_EVALUATOR_EFFORT = "medium"

# Fallback khi Java khong gui thang xuong (goi tay, pipeline nghien cuu). Dung
# build_band_ladder() de dung tu du lieu that -- hang so nay khoa vao VSTEP 6 bac.
BAND_LADDER = """SIX-BAND SPEAKING LADDER - KEEP THIS PREFIX EXACT
1 BAC_1: concrete, immediate personal information; short simple descriptions.
2 BAC_2: familiar matters; connected basic details with limited reasons.
3 BAC_3: familiar and some less familiar matters; compare options and explain reasons.
4 BAC_4: develop a clear argument; handle abstraction and causal relationships.
5 BAC_5: flexible, precise discussion of complex or hypothetical implications.
6 BAC_6: nuanced synthesis, subtle distinctions, and well-controlled complex reasoning.
END SIX-BAND SPEAKING LADDER"""

SAFETY_CONSTRAINTS = """Reject any question that assumes overseas travel, family structure,
economic resources, device ownership, specialist knowledge, politics, religion, or regional
stereotypes. The prompt must be neutral and answerable by every Vietnamese high-school student."""

TOPICS = [
    ("School clubs", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Healthy routines", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Films and stories", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Games and technology", "TECH_GAMING", "OUT_OF_CURRICULUM"),
    ("Places in my town", "TRAVEL_PLACES", "IN_GDPT2018"),
    ("Everyday science", "FUTURE_SCIENCE", "IN_GDPT2018"),
    ("Learning with friends", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Sports choices", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Music and performance", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Future inventions", "FUTURE_SCIENCE", "IN_GDPT2018"),
]

CRITERIA = [
    ("PRONUNCIATION", None),
    ("FLUENCY", None),
    ("VOCABULARY", None),
    ("GRAMMAR", "tense_control"),
    ("GRAMMAR", "complex_clause_control"),
    ("COHERENCE", "weak_progression"),
    ("COHERENCE", "limited_support"),
]

# Taxonomy dong, 4 nhan. Truoc day 13.
#
# Phep thu de giu mot nhan: nhan do lai duoc can gat nao khi ra de? Bon nhan duoi day moi
# nhan ung voi dung mot can gat co that -- khung thoi gian cau hoi, kieu lap luan, dang cau
# hoi. Chin nhan bi cat deu DO duoc, nhung khong co cach nao ra mot de nham trung chung.
#
# PHAI khop tung chu voi SubAttributePolicy.java ben Java: taxonomy nay duoc khai bao lap o
# 4 noi (policy Java, hang so nay, prompt cham, prompt sinh). Lech mot noi thi nhan bi loc
# LANG LE, khong no.
ALLOWED_SUB_ATTRIBUTES: dict[str, frozenset[str | None]] = {
    "PRONUNCIATION": frozenset({None}),
    "FLUENCY": frozenset({None}),
    # VOCABULARY khong con nhan nao -- chi luyen duoc o muc tieu chi.
    "VOCABULARY": frozenset({None}),
    "GRAMMAR": frozenset({"tense_control", "complex_clause_control"}),
    "COHERENCE": frozenset({"weak_progression", "limited_support"}),
}

FILTER_REASON_CODES = frozenset(
    {
        "NOT_ENGLISH",
        "LENGTH_OUT_OF_RANGE",
        "MISSING_FIELD",
        "SUB_ATTRIBUTE_NOT_ALLOWED",
        "CRITERION_UNKNOWN",
        "DUPLICATE_COSINE",
    }
)


def build_band_ladder(rungs, band_count: int = 6) -> str:
    """Dung mo ta thang bac tu du lieu Java gui xuong.

    Vi sao khong giu hang so BAND_LADDER: no viet cung BAC_1..BAC_6 kieu VSTEP, doi truong
    sang CEFR/IELTS thi prompt cham mo ta SAI thang ma khong ai biet. Rong -> lui ve hang so
    cu de goi tay/pipeline nghien cuu van chay nhu truoc.
    """
    if not rungs:
        return BAND_LADDER
    header = f"{band_count}-BAND SPEAKING LADDER - KEEP THIS PREFIX EXACT"
    lines = []
    for rung in sorted(rungs, key=lambda item: _rung_field(item, "order", 0)):
        order = _rung_field(rung, "order", 0)
        code = _rung_field(rung, "code", "")
        description = _rung_field(rung, "description", "")
        lines.append(f"{order} {code}: {description}".rstrip(": "))
    footer = f"END {band_count}-BAND SPEAKING LADDER"
    return "\n".join([header, *lines, footer])


def _rung_field(rung, name: str, default):
    if isinstance(rung, dict):
        return rung.get(name, default)
    return getattr(rung, name, default)
