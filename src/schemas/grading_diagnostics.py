from pydantic import BaseModel, Field


class GradingDiagnosticsItemRequest(BaseModel):
    item_id: str
    feedback_summary: str = Field(min_length=1)
    # Ma tieu chi giao vien da cham cho cau nay (vd "grammar", "vocabulary", "pronunciation").
    # Chi tieu chi nao co trong ALLOWED_WEAKNESS_LABELS (utils/criterion_diagnostics.py) moi duoc
    # suy luan nhan -- pronunciation/fluency van vao tu tin hieu audio, khong co taxonomy text.
    criteria: list[str] = Field(default_factory=list)


class GradingDiagnosticsRequest(BaseModel):
    items: list[GradingDiagnosticsItemRequest] = Field(default_factory=list)


class GradingDiagnosticsLabel(BaseModel):
    criterion_code: str
    label: str
    evidence_span: str = ""


class GradingDiagnosticsItemResult(BaseModel):
    item_id: str
    labels: list[GradingDiagnosticsLabel] = Field(default_factory=list)


class GradingDiagnosticsResponse(BaseModel):
    items: list[GradingDiagnosticsItemResult] = Field(default_factory=list)
