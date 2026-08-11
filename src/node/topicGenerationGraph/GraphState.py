from typing import TypedDict

from schemas.topic_generation import (
    TopicProposal,
    TopicProposalRequest,
)


class TopicGenerationState(TypedDict, total=False):
    request: TopicProposalRequest

    # De xuat cua RIENG vong hien tai -- dau vao cua topic_dedupe. Bi ghi de moi vong.
    proposals: list[TopicProposal]

    # Cai da qua duoc bo loc, TICH LUY qua cac vong. Day moi la ket qua tra ve.
    accepted: list[TopicProposal]

    # Embedding cua `accepted`, giu lai de vong sau con so duoc voi vong truoc.
    #
    # Phai nam trong state chu khong the tinh lai tu Chroma: chu de vua duoc chap nhan CHUA
    # duoc index -- Java moi goi generationClient.index() sau khi tao xong. Khong giu thi hai
    # vong lien tiep co the tra ve cung mot chu de.
    accepted_embeddings: list[list[float]]

    # Ten chu de ma de xuat vua va cham phai -- dua nguoc vao prompt cua vong sau.
    collisions: list[str]

    round: int
