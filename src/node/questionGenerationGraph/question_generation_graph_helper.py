from __future__ import annotations

import math
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from node.questionGenerationGraph.constants import (
    EMBEDDING_MODEL,
    MODEL,
)
from schemas.question_generation import (
    CandidateVerdict,
    PracticeQuestionCandidate,
    difficulty_rank,
)
from vector.chroma_client import build_raw_collection


@dataclass
class TokenCall:
    role: str
    mode: str
    model: str
    input: int
    output: int
    reasoning: int
    cached_input: int
    response_id: str


class QuestionGenerationRuntime:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.question_collection = build_raw_collection(
            "practice_questions",
            embedding_model=EMBEDDING_MODEL,
        )
        self.topic_collection = build_raw_collection(
            "practice_topics",
            embedding_model=EMBEDDING_MODEL,
        )

    def parsed_call(
        self,
        *,
        role: str,
        mode: str,
        effort: str,
        system: str,
        prompt: str,
        schema: Any,
        tokens: list[TokenCall],
    ) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter()
        response = self.client.responses.parse(
            model=MODEL,
            reasoning={"effort": effort},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            text_format=schema,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"{role} returned no structured output")
        usage = response.usage
        input_details = usage.input_tokens_details if usage else None
        output_details = usage.output_tokens_details if usage else None
        call = TokenCall(
            role=role,
            mode=mode,
            model=MODEL,
            input=usage.input_tokens if usage else 0,
            output=usage.output_tokens if usage else 0,
            reasoning=output_details.reasoning_tokens if output_details else 0,
            cached_input=input_details.cached_tokens if input_details else 0,
            response_id=response.id,
        )
        tokens.append(call)
        return response.output_parsed, {
            "response_id": response.id,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "parsed": response.output_parsed.model_dump(),
            "tokens": asdict(call),
        }

    def embed(self, text: str) -> tuple[list[float], int]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding, response.usage.prompt_tokens

    def max_similarity(
        self,
        embedding: list[float],
        exclude_ids: set[str] | None = None,
    ) -> float:
        """Cau da co trong kho giong ban nhap nay nhat la bao nhieu -- BO QUA nhung cau da chet
        vinh vien voi hoc sinh dang cho.

        Vi sao phai bo qua: Chroma khong co khai niem hoc sinh (where chi loc {"active": True}),
        con luat loai cau thi theo TUNG hoc sinh. Hai pham vi lech nhau tao ra khoa cung:

            HS luyen cau A, dat bac  -> A loai vinh vien voi em ay
            chon lai chu de do       -> doc kho khong con gi
            nho LLM soan cau moi     -> cung chu de/tieu chi/bac nen rat giong A
            chan trung 0,92 voi A    -> vut sach -> pool_exhausted, MAI MAI

        Thu chan no lai la thu no khong duoc phep dung. Loai A ra khoi phep so thi ban nhap moi
        song, va hoc sinh co cau de luyen tiep.

        Danh doi da biet: kho SE co hai cau gan trung nhau (A va ban moi). Voi hoc sinh nay thi
        vo hai -- A da chet. Voi hoc sinh khac chua gap ca hai thi co the gap lan luot qua nhieu
        buoi; cong chan-hoi trong mot phien (max_similarities, 0,85) van do duoc phan trong cung
        mot buoi. Chap nhan, vi doi lai la go duoc khoa cung.

        Chroma khong loc duoc theo id trong query(), nen lay du n_results roi bo -- lay
        len(exclude) + 5 de con du ung vien sau khi bo het phan bi loai.
        """
        total = self.question_collection.count()
        if total == 0:
            return 0.0
        excluded = exclude_ids or set()
        n_results = min(total, len(excluded) + 5)
        result = self.question_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"active": True},
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for question_id, distance in zip(ids, distances, strict=True):
            if question_id in excluded:
                continue
            # Chroma tra ve theo thu tu khoang cach tang dan, nen cai dau tien khong bi loai
            # CHINH LA cai giong nhat -- khong can duyet het.
            return max(0.0, min(1.0, 1.0 - float(distance)))
        return 0.0


def question_embedding_text(topic_name: str, prompt_text: str) -> str:
    return f"Topic: {topic_name}\nQuestion: {prompt_text}"


def stable_topic_id(name: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"vox-practice-topic:{normalize_name(name)}",
        )
    )


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        value.casefold().strip().replace("đ", "d"),
    )
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.split())


def verdict_signature(
    verdict: CandidateVerdict,
) -> tuple[bool, tuple[str, ...]]:
    return verdict.accepted, tuple(sorted(verdict.violations))


def question_record(
    *,
    question_id: str,
    topic: tuple[str, str, str],
    candidate: PracticeQuestionCandidate,
    topic_id: str | None = None,
    band_count: int = 6,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "candidate_id": candidate.candidate_id,
        "topic_id": topic_id or stable_topic_id(topic[0]),
        "topic_name": topic[0],
        "interest_dimension": topic[1],
        "curriculum_group": topic[2],
        "question_text": candidate.prompt_text,
        "target_criterion_code": candidate.target_construct,
        "target_sub_attribute": candidate.target_sub_attribute,
        # Ghi vao ban ghi de Java luu xuong practice_question.target_tense. Thieu doan nay thi
        # thi chi ton tai trong luc soan roi bien mat, va thang leo khong bao gio tra duoc
        # "cau qua khu cua chu de nay" -- moi luot doi thi lai phai goi LLM.
        "target_tense": candidate.target_tense,
        "difficulty_rank": difficulty_rank(candidate.difficulty_features, band_count),
        "difficulty_features": candidate.difficulty_features.model_dump(),
        "evaluation_guide": candidate.evaluation_guide.model_dump(),
        "suggested_ideas": candidate.suggested_ideas,
        "question_type": candidate.question_type,
        "min_response_seconds": candidate.min_response_seconds,
        "max_response_seconds": candidate.max_response_seconds,
        "vstep_part": candidate.vstep_part,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def rank_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    # Suy khoang tu chinh du lieu, KHONG cung range(1, 7): thang bac do framework quyet dinh
    # (IELTS 9 bac), rank >= 7 tung lam ham nay nem KeyError.
    ranks = [int(record["difficulty_rank"]) for record in records]
    if not ranks:
        return {}
    result = {str(rank): 0 for rank in range(1, max(ranks) + 1)}
    for rank in ranks:
        result[str(rank)] += 1
    return result
