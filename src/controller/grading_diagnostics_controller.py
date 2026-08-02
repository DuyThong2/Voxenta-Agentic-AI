from fastapi import APIRouter

from node.gradingDiagnosticsGraph.service import infer_grading_diagnostics
from schemas.grading_diagnostics import GradingDiagnosticsRequest, GradingDiagnosticsResponse

router = APIRouter(prefix="/internal/grading-diagnostics", tags=["Grading diagnostics"])


@router.post("/infer", response_model=GradingDiagnosticsResponse)
async def infer_grading_diagnostics_endpoint(
    request: GradingDiagnosticsRequest,
) -> GradingDiagnosticsResponse:
    return infer_grading_diagnostics(request)
