from schemas.topic_generation import TopicProposalRequest


def build_topic_proposal_prompt(request: TopicProposalRequest) -> str:
    # Không có từ khoá nào thì luật "tối đa MỘT chủ đề được vượt ra ngoài từ khoá quan sát
    # được" tự mâu thuẫn: mọi chủ đề đều nằm ngoài. Nói thẳng cho model biết bằng chứng lúc
    # này là interest_scores, thay vì để nó tự đoán cách dung hoà một luật bất khả thi.
    grounding_rule = (
        "3. At most ONE topic may go beyond observed keywords; set grounded_in_keyword=false."
        if request.keyword_evidence
        else "3. There are no observed keywords. Base every topic on the interest scores: set "
             "grounded_in_keyword=false and evidence_type=INTEREST on all of them."
    )
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
{grounding_rule}
4. Set evidence_type to KEYWORD, INTEREST, EXHAUSTED, or SEARCH. For KEYWORD and
   SEARCH, list only supporting input keywords in evidence_keywords.
5. Confidence must follow evidence: one session <=0.5, two <=0.7, three or more
   <=0.85; INTEREST <=0.6; EXHAUSTED <=0.7; ungrounded <=0.4; SEARCH <=0.95.
6. Every topic must differ meaningfully from pool and rejected topics. Explain in
   distinct_from.
7. Keep topics inclusive and answerable without money, travel, or specialist knowledge.
8. For direct search, return one proposal only, evidence_type=SEARCH, and confidence
   must be >=0.9 or return none.
9. Set temporal_affordance to the time frame the topic itself pulls toward. This decides
   which tense later questions will drill, so answer about the TOPIC, not about a single
   question you imagine:
   - PAST: the subject is finished or historical, and most natural questions look
     backwards ("The history of my school", "A festival my class held").
   - FUTURE: the subject has not happened yet ("Inventions that could change school",
     "Where I want to study after graduation").
   - MIXED: everything else, including anything discussable in more than one time frame.
     Choose MIXED when unsure - it lets the system rotate tenses across a session, whereas
     a wrong PAST/FUTURE locks every question of this topic into the wrong time frame.

Return the best proposals, not diverse samples. Structured data only."""
