from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field, model_validator

# Dinh nghia O DAY chu khong o node/questionGenerationGraph/constants.py: schemas la tang duoi,
# node import len duoc con nguoc lai thi vong lap (constants -> package __init__ -> cac node ->
# schemas). constants.py re-export lai de moi cho van goi ten quen thuoc.
#
# 5 chu khong 3: tu khi bo cong chan-trung-lich-su o CandidateFilterNode, ung vien chi con rot
# o luat cung (do dai, ASCII, taxonomy) va o evaluator. Soan du ra vai cai de mot lo hong bao
# gio ve tay khong -- lo rong nghia la pool_exhausted, tuc phien luyen dut giua chung. Chi phi
# them nam trong CUNG mot luot drafter, khong phai them vong nao.
DRAFTER_CANDIDATES = 5

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

# Tap dong 5 thi. PHAI khop tung chu voi TensePolicy.java -- lech mot noi thi cau bi loc LANG
# LE o CandidateFilterNode, khong no. Day dung cai bay da dinh voi taxonomy sub-attribute.
#
# San do kho cua tung thi (raw_difficulty): PRESENT 1; PAST/FUTURE/PERFECT 2 (khong neo vao
# hien tai -> here_and_now=False -> +1); CONDITIONAL 4 (them reasoning_type=hypothetical -> +2).
# Java da chan truoc bang TensePolicy.allowedFor nen o day khong lap lai phep chan do.
Tense = Literal["PRESENT", "PAST", "FUTURE", "PERFECT", "CONDITIONAL"]

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
    # Thi ma cau nay EP hoc sinh dung. Bat buoc o duong sinh: de None duoc thi mo hinh se de
    # trong bat cu khi nao no thay kho, va ca co che ep thi thanh tuy chon.
    target_tense: Tense
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
    # Bam theo DRAFTER_CANDIDATES (5). Khong dat cung so o hai noi: prompt bao "generate exactly
    # N" con schema ep dung N, lech nhau la LLM tra ve hop le ma pydantic van nem.
    candidates: list[PracticeQuestionCandidate] = Field(
        min_length=DRAFTER_CANDIDATES,
        max_length=DRAFTER_CANDIDATES,
    )

    @model_validator(mode="after")
    def unique_ids(self) -> "DraftBatch":
        if len({candidate.candidate_id for candidate in self.candidates}) != DRAFTER_CANDIDATES:
            raise ValueError("Drafter candidate IDs must be unique")
        return self


class CandidateVerdict(BaseModel):
    candidate_id: str
    accepted: bool
    violations: list[str]


class EvaluationBatch(BaseModel):
    verdicts: list[CandidateVerdict]


class RefinedBatch(BaseModel):
    # Tran bam theo DRAFTER_CANDIDATES: refiner khong the tra ve nhieu hon so ung vien di vao.
    # De cung 4 trong khi drafter sinh 5 thi mot lo toan cau tot lai bi pydantic nem.
    candidates: list[PracticeQuestionCandidate] = Field(
        min_length=1,
        max_length=DRAFTER_CANDIDATES,
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
    # Java gui xuong thi dich cho O nay (TensePolicy.forSlot). None = goi tay/pipeline nghien
    # cuu, luc do de mo hinh tu chon thi tu nhien nhat cho chu de.
    target_tense: Tense | None = None
    # le=20 chu khong 6: thang bac do framework cua truong quyet dinh (CEFR 6, IELTS 9...),
    # khong phai hang so VSTEP. Tran that do Java tinh tu framework_result_bands.
    target_rank: int = Field(ge=1, le=20)
    # le=5 theo DRAFTER_CANDIDATES: Java xin bao nhieu cau thi nhieu nhat cung chi bang so
    # ung vien mot luot drafter sinh ra.
    count: int = Field(default=3, ge=1, le=DRAFTER_CANDIDATES)
    # So bac cua thang dang ap; Java gui xuong. Mac dinh 6 de goi tay/test khong vo.
    band_count: int = Field(default=6, ge=1, le=20)
    # Mo ta tung bac, de dung ladder trong prompt cham. Rong -> dung BAND_LADDER mac dinh.
    band_ladder: list[BandRung] = Field(default_factory=list)
    # Cau da CHET VINH VIEN voi chinh hoc sinh dang cho: bi loai khoi phep so trung o
    # CandidateFilterNode. Xem runtime.max_similarity de biet vi sao.
    #
    # Rong (goi tay, pipeline nghien cuu, hoac hoc sinh chua luyen cau nao) -> so voi ca kho
    # nhu cu, khong doi hanh vi.
    exclude_question_ids: list[str] = Field(default_factory=list)


class GeneratedQuestion(BaseModel):
    id: str
    topic_id: str
    topic_name: str
    question_text: str
    target_criterion_code: CriterionCode
    target_sub_attribute: str | None
    target_tense: Tense | None
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


def raw_for_rank(rank: int, band_count: int = RAW_DIFFICULTY_MAX) -> int:
    """Nghịch đảo của `difficulty_rank`: bậc của trường -> mức khó thô Robinson 1..6.

    Cần vì nút soạn câu phải biết đặt bốn cần gạt Robinson ở đâu, mà bốn cần gạt đó chỉ nói
    chuyện bằng thang thô. Với thang 6 bậc thì đây là phép đồng nhất.
    """
    safe_band_count = max(1, int(band_count))
    safe_rank = max(1, min(safe_band_count, int(rank)))
    if safe_band_count == RAW_DIFFICULTY_MAX:
        return safe_rank
    if safe_band_count == 1:
        return RAW_DIFFICULTY_MIN
    normalized = (safe_rank - 1) / (safe_band_count - 1)
    raw = RAW_DIFFICULTY_MIN + round(
        normalized * (RAW_DIFFICULTY_MAX - RAW_DIFFICULTY_MIN)
    )
    return max(RAW_DIFFICULTY_MIN, min(RAW_DIFFICULTY_MAX, int(raw)))


def feature_profiles_for_raw(raw: int) -> list[DifficultyFeatures]:
    """Mọi tổ hợp đặc trưng Robinson cho ra ĐÚNG mức khó thô này.

    Liệt kê vét cạn từ chính `raw_difficulty` chứ không viết tay: đổi trọng số trong công
    thức thì danh sách này tự đi theo, không có cách nào lệch.

    Vì sao cần: `raw_difficulty` là phép ĐO trên bốn đặc trưng, không phải cái đích nhắm
    được. Bảo mô hình "viết câu bậc 6" giống bảo ai đó "hãy cao 1m80" -- cao là kết quả.
    Thứ nhắm được là bốn cần gạt. Và khoảng tự do rất lệch nhau:

        bậc 1: 2/60 tổ hợp      bậc 4: 19/60
        bậc 2: 9/60             bậc 5: 10/60
        bậc 3: 18/60            bậc 6: 2/60

    Hai đầu thang gần như chỉ có một cách viết đúng. Không nói ra thì xác suất trúng bậc 6
    khi soạn tự do chỉ ~3%, và đó chính là lý do kho không bao giờ tự có câu khó.

    Thứ tự trả về ổn định, ưu tiên tổ hợp tự nhiên trước: cái trừu tượng thì thường cũng
    không phải chuyện đang diễn ra trước mắt, nên xếp `abstract` + `here_and_now=False` lên
    trên. Chỉ là thứ tự ưu tiên, không phải điểm số.
    """
    safe_raw = max(RAW_DIFFICULTY_MIN, min(RAW_DIFFICULTY_MAX, int(raw)))
    profiles: list[DifficultyFeatures] = []
    for here_and_now in (True, False):
        # 2 và 5 chỉ là đại diện hai phía ngưỡng `num_elements >= 4`; công thức không phân
        # biệt gì hơn trong mỗi phía.
        for num_elements in (2, 5):
            for reasoning_type in get_args(ReasoningType):
                for abstractness in get_args(Abstractness):
                    features = DifficultyFeatures(
                        here_and_now=here_and_now,
                        num_elements=num_elements,
                        reasoning_type=reasoning_type,
                        abstractness=abstractness,
                    )
                    if raw_difficulty(features) == safe_raw:
                        profiles.append(features)
    profiles.sort(
        key=lambda item: (
            item.abstractness == "abstract" and item.here_and_now,
            # `mixed` không đổi được mức khó (chỉ `abstract` mới cộng 1) nên để sau -- nói
            # "mixed" với người soạn câu là một chỉ dẫn mơ hồ không mua được gì.
            item.abstractness == "mixed",
        )
    )
    return profiles


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
