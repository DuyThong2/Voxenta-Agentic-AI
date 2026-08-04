import json
from typing import Any, Dict, List


def _band_ladder_block(bands: List[Any]) -> str:
    lines = []
    for band in sorted(bands, key=lambda b: getattr(b, "order", 0)):
        descriptor = (getattr(band, "descriptor", None) or "").strip()
        label = getattr(band, "label", None) or band.code
        lines.append(f"  - {band.code} ({label}): {descriptor or '(no descriptor)'}")
    return "\n".join(lines) or "  (no bands configured)"


def build_measured_band_match_prompt(entries: List[Dict[str, Any]]) -> str:
    """Ghép bằng chứng ĐO ĐƯỢC của từng tiêu chí với thang mô tả bậc của chính tiêu chí đó.

    Chỉ dùng cho các tiêu chí do máy chấm (phát âm, độ trôi chảy). Chúng không đi qua
    LanguageQualityEvalNode -- node duy nhất có phần khớp bậc -- nên trước đây
    matched_band_code của chúng luôn rỗng, và mọi bản ghi phát âm/trôi chảy đều bị loại khỏi
    phép ước lượng bậc trong im lặng.
    """
    blocks = []
    for entry in entries:
        metrics = json.dumps(entry["metrics"], ensure_ascii=False)
        blocks.append(
            f"### {entry['criterion_key']}\n"
            f"Measured evidence (Azure speech assessment, 0-100 unless noted): {metrics}\n"
            f"Band ladder for this criterion:\n{_band_ladder_block(entry['bands'])}"
        )
    joined = "\n\n".join(blocks)
    keys = ", ".join(entry["criterion_key"] for entry in entries)
    return f"""You are placing a learner's measured speaking performance onto a band ladder.

{joined}

For each criterion above, decide which band's descriptor the performance ACTUALLY matches best.

Rules:
- Judge against the DESCRIPTOR text, not against the raw number. The descriptors talk about
  observable behaviour (hesitation, pausing, intelligibility, mother-tongue influence); the
  measured evidence is what those behaviours look like as numbers.
- The measured scores come from an automatic speech assessment that is known to be generous.
  A high accuracy number does not by itself mean a high band -- check whether the described
  behaviour actually fits.
- This judgement is INDEPENDENT of any target band. Return a weaker or stronger band whenever
  the evidence points there.
- Use only band codes listed in that criterion's ladder above. Never invent a code, never
  reformat one (BAC_3, not "Bậc 3" or "bac_3").
- If the evidence is too thin to place a criterion, return an empty string for it. An empty
  answer is treated as "unknown" and dropped; a guessed band would be counted as real.

Return only a JSON object, no prose, with exactly this shape ({keys}):
{{"matches": {{"<criterion_key>": {{"band_code": "<CODE or empty string>", "reason": "<one short sentence>"}}}}}}"""
