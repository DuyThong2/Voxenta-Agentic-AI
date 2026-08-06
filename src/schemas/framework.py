import math
from typing import List, Optional

from pydantic import Field, model_validator

from schemas.common import _CamelMessage


class FrameworkBand(_CamelMessage):
    code: str
    label: Optional[str] = None
    score_min: float
    score_max: float
    descriptor: Optional[str] = None
    positive_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)
    order: int = 0


# Khoá tiêu chí mà các nút chấm thật sự tra cứu. Xem _find_framework ở
# LanguageQualityEvalNode / AzureScoreScaleNode / PronunciationNode -- cả ba so khớp TUYỆT ĐỐI
# với đúng những chuỗi này.
AGENT_CRITERION_KEYS = frozenset(
    {"pronunciation", "fluency", "grammar", "vocabulary", "coherence"}
)


def _normalize_criterion_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    # "discourse" là tên cũ của cùng tiêu chí; giữ phép quy đổi đã có từ trước.
    return "coherence" if normalized == "discourse" else normalized


class CriterionFramework(_CamelMessage):
    # Plain str, not the CriterionName Literal: this is authored on the Java
    # side and forwarded as-is. An unrecognized key must not fail parsing of
    # the whole event — it simply won't match any node's lookup downstream
    # (see each eval node's *_node_helper.build_framework_criterion_context).
    criterion_key: str
    # Khoá NGUYÊN VĂN Java gửi xuống, giữ lại trước khi resolve_criterion_key chuẩn hoá.
    # Dùng để TRẢ VỀ đúng từ vựng của Java -- xem source_key_for / to_source_keys.
    source_criterion_key: str = ""
    framework_code: Optional[str] = None
    framework_criterion_name: Optional[str] = None
    framework_criterion_description: Optional[str] = None
    target_band_id: Optional[str] = None
    target_band_code: Optional[str] = None
    target_band_label: Optional[str] = None
    target_band_only: bool = False
    rubric_weight: Optional[float] = None
    rubric_min_score: float = 0
    rubric_max_score: float = 100
    bands: List[FrameworkBand] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolve_criterion_key(self):
        """Đưa criterion_key về đúng tên tiêu chí chuẩn -- KEY TRƯỚC, không khớp mới tra tiếp.

        Java lấy khoá từ ``rubric_criterions.code``, tức GIẢ ĐỊNH mã rubric chính là tên tiêu
        chí chuẩn. Đúng với rubric đặt mã ``grammar``/``vocabulary``..., nhưng trường được tự
        đặt mã: "Rubric 2026" dùng ``TC01..TC05`` nên khoá gửi xuống là ``tc01..tc05``.

        Đo trên bài chấm 2026-08-06: không khoá nào khớp -> ``_find_framework`` trả None -> mô
        hình được bảo "chấm 0-100" thay vì theo mô tả bậc của trường -> nó tự bịa mã bậc
        (``V_21_40``, ``weak``) -> bị lọc sạch -> ``matched_band_code`` luôn NULL. Cả chuỗi chỉ
        hiện ra bằng một dòng WARNING.

        Sửa ở ĐÂY chứ không ở từng nút: ba nơi cùng tra bằng so khớp tuyệt đối
        (LanguageQualityEval, AzureScoreScale, Pronunciation), vá riêng lẻ là mở đường cho
        chúng lệch nhau. Chuẩn hoá ngay lúc parse thì cả ba tự đúng.

        Thứ tự tin cậy giảm dần:
          1. ``criterion_key``            -- giữ nguyên hành vi cũ khi Java đã gửi đúng
          2. ``framework_code``           -- danh mục chuẩn (PRONUNCIATION/GRAMMAR/...)
          3. ``framework_criterion_name`` -- vớt nốt; thực tế thường là tiếng Việt ("Ngữ pháp")
             nên hiếm khi trúng, giữ lại cho trường đặt tên tiêu chí bằng tiếng Anh

        Không nguồn nào trúng thì GIỮ NGUYÊN khoá gốc: gửi khoá lạ vẫn hơn bịa, và các nút đã
        chịu được khoá không khớp (mất phần mô tả bậc, điểm vẫn chấm).
        """
        if not self.source_criterion_key:
            self.source_criterion_key = self.criterion_key
        for candidate in (
            self.criterion_key,
            self.framework_code,
            self.framework_criterion_name,
        ):
            resolved = _normalize_criterion_key(candidate)
            if resolved in AGENT_CRITERION_KEYS:
                self.criterion_key = resolved
                return self
        return self

    @model_validator(mode="after")
    def validate_target_band_scope(self):
        if (
            not math.isfinite(self.rubric_min_score)
            or not math.isfinite(self.rubric_max_score)
            or self.rubric_max_score <= self.rubric_min_score
        ):
            raise ValueError(
                "rubric_max_score must be greater than rubric_min_score"
            )
        if not self.target_band_only:
            return self
        if not self.target_band_code:
            raise ValueError("target_band_code is required when target_band_only is true")
        if not self.bands:
            raise ValueError(
                "target-band-only scoring must include at least the target framework band"
            )
        target_band = next(
            (band for band in self.bands if band.code == self.target_band_code),
            None,
        )
        if target_band is None:
            raise ValueError("bands must include an entry matching target_band_code")
        if (
            target_band.score_min != self.rubric_min_score
            or target_band.score_max != self.rubric_max_score
        ):
            raise ValueError(
                "the target band must cover the complete rubric score range"
            )
        return self


def to_source_keys(
    criteria: dict,
    criteria_frameworks: List[CriterionFramework] | None,
) -> dict:
    """Đổi khoá tiêu chí về đúng từ vựng Java đã gửi xuống, trước khi trả kết quả.

    Bên trong đồ thị chấm, mọi thứ dùng khoá chuẩn (``grammar``, ``vocabulary``...) vì các nút
    tra cứu bằng đúng những tên đó. Nhưng Java lập chỉ mục rubric bằng CHÍNH MÃ NÓ GỬI ĐI
    (``rubric_criterions.code``), nên trả về khoá chuẩn là nó không tra ra gì.

    Đo trên bài chấm 2026-08-06: rubric "Rubric 2026" đặt mã ``TC01..TC05``; Python trả về
    ``pronunciation/fluency/grammar/vocabulary/coherence``; ``computeItemScore`` không khớp
    tiêu chí nào nên điểm ra 0.00, rồi 0.00 không thuộc dải điểm kết quả nào -> ném
    IllegalStateException -> DLT -> phiên bị đánh GRADING_FAILED. Tức bài chấm xong xuôi vẫn
    hiện "chấm lỗi".

    Nguyên tắc: bên nào đặt tên thì bên đó là chủ của tên. Python chuẩn hoá để tra cứu NỘI BỘ,
    nhưng nói lại đúng từ vựng đã nhận khi trả lời.

    Tiêu chí không có framework tương ứng thì giữ nguyên khoá chuẩn -- Java bỏ qua khoá lạ
    (``criterion == null -> continue``) đúng như trước, không tệ hơn.
    """
    if not criteria_frameworks:
        return criteria
    source_by_canonical = {
        framework.criterion_key: framework.source_criterion_key
        for framework in criteria_frameworks
        if framework.source_criterion_key
    }
    return {
        source_by_canonical.get(name, name): score
        for name, score in criteria.items()
    }
