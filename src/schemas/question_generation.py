from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriterionCode = Literal[
    "PRONUNCIATION",
    "FLUENCY",
    "GRAMMAR",
    "VOCABULARY",
    "COHERENCE",
]
ReasoningType = Literal[
    "description",
    "comparison",
    "causal",
    "intentional",
    "hypothetical",
]
Abstractness = Literal["concrete_personal", "mixed", "abstract"]

# KHONG co READ_ALOUD: dang do can van ban mau de doc theo, ma luyen noi tu do thi khong
# co van ban nao. Bon dang con lai dung bang enum QuestionType cua de thi.
PracticeQuestionType = Literal[
    "SHORT_ANSWER",
    "LONG_ANSWER",
    "DESCRIPTION",
    "OPINION",
]

# Dai thoi luong cho phep theo tung dang: (san_thap, san_cao), (tran_thap, tran_cao).
#
# Lay theo TI LE ~60-70% cua de thi that (SHORT_ANSWER 20-30/45-60, LONG_ANSWER 30/75-90,
# DESCRIPTION 45-60/90-150, OPINION 45/120) chu khong be nguyen. Ly do: ngan sach mot
# phien luyen hien la 300 giay, mot cau DESCRIPTION 150 giay an het nua phien -- ca buoi
# duoc 2 cau. Tinh than luyen tap la NHIEU LUOT NGAN lap nhieu, khac han thi (mot lan,
# dai, diem cao). Voi dai co lai thi 300 giay ra 4-8 cau moi buoi.
#
# Doi chinh sach thi sua DUY NHAT bang nay.
PRACTICE_TYPE_SECONDS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    # Sàn 15-25s mâu thuẫn với chính định nghĩa của loại này ("a couple of sentences is a
    # complete answer" -- vài câu ở tốc độ nói bình thường là ~10-15s). Model giải mâu thuẫn
    # đó bằng cách gộp thêm vế hỏi cho đủ giây, nên SHORT_ANSWER sinh ra toàn câu hai vế.
    # Hạ sàn để con số thôi ép ngược lại luật "một vế" trong drafter_prompt.
    "SHORT_ANSWER": ((10, 18), (30, 45)),
    "LONG_ANSWER": ((25, 35), (50, 65)),
    "DESCRIPTION": ((30, 45), (60, 80)),
    "OPINION": ((35, 45), (60, 85)),
}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


class DifficultyFeatures(BaseModel):
    here_and_now: bool
    num_elements: int = Field(ge=1, le=8)
    reasoning_type: ReasoningType
    abstractness: Abstractness


class EvaluationGuide(BaseModel):
    expected_content: str = Field(min_length=1)
    key_points: str = Field(min_length=1)
    acceptable_responses: str = Field(min_length=1)
    off_topic_examples: str = Field(min_length=1)
    scoring_hints: str = Field(min_length=1)
    common_mistakes: str = Field(min_length=1)


class PracticeQuestionCandidate(BaseModel):
    candidate_id: str
    difficulty_features: DifficultyFeatures
    target_construct: CriterionCode
    target_sub_attribute: str | None = Field(default=None, max_length=64)
    vstep_part: int = Field(ge=1, le=3)
    question_type: PracticeQuestionType
    prompt_text: str = Field(min_length=1)
    suggested_ideas: list[str] = Field(min_length=2, max_length=4)
    # SAN va TRAN, dung hinh dang cua cau hoi de thi (bang questions). Khong con
    # planning_time_seconds (luyen tap bam-de-di-tiep, hoc sinh tu quyet luc nao san sang)
    # cung khong con max_followup_seconds (follow-up da duoc cong don giay chung voi cau
    # chinh roi, khong can ngan sach rieng) -- xem migration V11.
    min_response_seconds: int = Field(gt=0, le=200)
    max_response_seconds: int = Field(gt=0, le=300)
    evaluation_guide: EvaluationGuide

    @model_validator(mode="after")
    def _clamp_to_type_range(self) -> "PracticeQuestionCandidate":
        """Kep thoi luong vao dai cua DANG BAI, roi ep san < tran.

        LLM de xuat ca dang lan thoi luong; ta khong ap dat con so nhung cung khong de no
        ra ngoai khuon. Khong kep thi mot cau SHORT_ANSWER 200 giay se an sach ngan sach
        phien, va DB con rang buoc max > min (V11) nen tra nguoc thu tu la INSERT no tan
        day sau khi da tieu het tien sinh cau hoi.

        Ep san bang cach HA SAN chu khong keo tran: tran da bi dai cua dang chan tren roi,
        keo len nua la pha chinh cai khuon vua kep xong.
        """
        min_range, max_range = PRACTICE_TYPE_SECONDS[self.question_type]
        self.min_response_seconds = _clamp(self.min_response_seconds, *min_range)
        self.max_response_seconds = _clamp(self.max_response_seconds, *max_range)
        if self.min_response_seconds >= self.max_response_seconds:
            self.min_response_seconds = max(1, self.max_response_seconds - 1)
        return self


class DraftBatch(BaseModel):
    candidates: list[PracticeQuestionCandidate] = Field(
        min_length=3,
        max_length=3,
    )

    @model_validator(mode="after")
    def unique_ids(self) -> "DraftBatch":
        if len({candidate.candidate_id for candidate in self.candidates}) != 3:
            raise ValueError("Drafter candidate IDs must be unique")
        return self


class CandidateVerdict(BaseModel):
    candidate_id: str
    accepted: bool
    violations: list[str]


class EvaluationBatch(BaseModel):
    verdicts: list[CandidateVerdict]


class RefinedBatch(BaseModel):
    candidates: list[PracticeQuestionCandidate] = Field(
        min_length=1,
        max_length=4,
    )


class BandRung(BaseModel):
    """Mot bac tren thang nang luc cua framework dang ap (Java gui xuong)."""

    order: int = Field(ge=1, le=20)
    code: str = ""
    description: str = ""


class QuestionGenerationRequest(BaseModel):
    topic_id: str
    topic_name: str = Field(min_length=1, max_length=200)
    interest_dimension: str = Field(min_length=1, max_length=32)
    curriculum_group: str = Field(min_length=1, max_length=24)
    target_criterion_code: CriterionCode
    target_sub_attribute: str | None = Field(default=None, max_length=64)
    # le=20 chu khong 6: thang bac do framework cua truong quyet dinh (CEFR 6, IELTS 9...),
    # khong phai hang so VSTEP. Tran that do Java tinh tu framework_result_bands.
    target_rank: int = Field(ge=1, le=20)
    count: int = Field(default=3, ge=1, le=3)
    # So bac cua thang dang ap; Java gui xuong. Mac dinh 6 de goi tay/test khong vo.
    band_count: int = Field(default=6, ge=1, le=20)
    # Mo ta tung bac, de dung ladder trong prompt cham. Rong -> dung BAND_LADDER mac dinh.
    band_ladder: list[BandRung] = Field(default_factory=list)


class GeneratedQuestion(BaseModel):
    id: str
    topic_id: str
    topic_name: str
    question_text: str
    target_criterion_code: CriterionCode
    target_sub_attribute: str | None
    difficulty_rank: int
    difficulty_features: dict
    evaluation_guide: dict
    suggested_ideas: list[str]
    question_type: str
    min_response_seconds: int
    max_response_seconds: int
    vstep_part: int


class QuestionGenerationResponse(BaseModel):
    questions: list[GeneratedQuestion]


class QuestionIndexRequest(BaseModel):
    question: GeneratedQuestion


# Khung Robinson chỉ định nghĩa từng đó chiều nên tổng luôn rơi vào 1..6 -- đây là số mức
# ĐẶC TRƯNG có thật, không phải chọn tuỳ tiện cho khớp VSTEP.
RAW_DIFFICULTY_MIN = 1
RAW_DIFFICULTY_MAX = 6


def raw_difficulty(features: DifficultyFeatures) -> int:
    """Mức độ khó thô theo Triadic Componential Framework (Robinson), luôn 1..6."""
    reasoning_weight = {
        "description": 0,
        "comparison": 1,
        "causal": 1,
        "intentional": 2,
        "hypothetical": 2,
    }[features.reasoning_type]
    raw = (
        1
        + (not features.here_and_now)
        + (features.num_elements >= 4)
        + reasoning_weight
        + (features.abstractness == "abstract")
    )
    return max(RAW_DIFFICULTY_MIN, min(RAW_DIFFICULTY_MAX, int(raw)))


def difficulty_rank(
    features: DifficultyFeatures,
    band_count: int = RAW_DIFFICULTY_MAX,
) -> int:
    """Ánh xạ mức khó thô sang thang bậc của framework đang áp.

    Vì sao ánh xạ chứ không sửa công thức: các chiều Robinson là nhị phân/tam phân nên chỉ
    sinh được 6 mức phân biệt. Với thang nhiều bậc hơn (IELTS 9), CHẤP NHẬN có bậc không bao
    giờ sinh ra được (N=9 -> chỉ ra {1,3,4,6,7,9}) thay vì nội suy -- nội suy là bịa độ chính
    xác không tồn tại. Thang leo chọn câu tự chịu được: bậc 1 tìm đúng rank, không có thì bậc
    2 nới ±1 nên vẫn bắt được câu lân cận.

    band_count = 6 trả về y hệt công thức cũ (bất biến không hồi quy).
    """
    raw = raw_difficulty(features)
    safe_band_count = max(1, int(band_count))
    if safe_band_count == RAW_DIFFICULTY_MAX:
        return raw
    if safe_band_count == 1:
        return 1
    normalized = (raw - RAW_DIFFICULTY_MIN) / (RAW_DIFFICULTY_MAX - RAW_DIFFICULTY_MIN)
    scaled = 1 + round(normalized * (safe_band_count - 1))
    return max(1, min(safe_band_count, int(scaled)))
