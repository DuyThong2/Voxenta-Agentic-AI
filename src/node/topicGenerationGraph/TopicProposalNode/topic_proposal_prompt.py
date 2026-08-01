from schemas.topic_generation import TopicProposalRequest


def build_topic_proposal_prompt(request: TopicProposalRequest) -> str:
    return f"""You are helping a Vietnamese high-school English speaking practice app decide
which NEW discussion topics to add for one specific learner.

Keyword evidence (counts are distinct sessions): {[
    item.model_dump() for item in request.keyword_evidence
]}
Interest scores: {request.interest_scores}
Topics already in the pool: {request.existing_topics}
Rejected topics: {request.rejected_topics}
Exhausted topics: {request.exhausted_topics}
This is a direct search keyword: {request.search_keyword}

Propose at most {request.max_proposals} new topics. Rules:
1. A topic must sustain a 10-15 minute spoken discussion for a B1-B2 learner.
2. Group related keywords into ONE topic. Do not echo a keyword as the topic.
3. At most ONE topic may go beyond observed keywords; set grounded_in_keyword=false.
4. Set evidence_type to KEYWORD, INTEREST, EXHAUSTED, or SEARCH. For KEYWORD and
   SEARCH, list only supporting input keywords in evidence_keywords.
5. Confidence must follow evidence: one session <=0.5, two <=0.7, three or more
   <=0.85; INTEREST <=0.6; EXHAUSTED <=0.7; ungrounded <=0.4; SEARCH <=0.95.
6. Every topic must differ meaningfully from pool and rejected topics. Explain in
   distinct_from.
7. Keep topics inclusive and answerable without money, travel, or specialist knowledge.
8. For direct search, return one proposal only, evidence_type=SEARCH, and confidence
   must be >=0.9 or return none.

Return the best proposals, not diverse samples. Structured data only."""
