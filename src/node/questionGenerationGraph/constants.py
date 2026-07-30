import os

from config.chroma_config import settings

MODEL = os.getenv("PRACTICE_GENERATION_MODEL", "gpt-5.4")
EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL
HARD_CAP = 80
DRAFTER_CANDIDATES = 3
MAX_EDITOR_ROUNDS = 3
REFINER_BATCH_SIZE = 4
DUPLICATE_THRESHOLD = 0.92

BAND_LADDER = """SIX-BAND SPEAKING LADDER - KEEP THIS PREFIX EXACT
1 BAC_1: concrete, immediate personal information; short simple descriptions.
2 BAC_2: familiar matters; connected basic details with limited reasons.
3 BAC_3: familiar and some less familiar matters; compare options and explain reasons.
4 BAC_4: develop a clear argument; handle abstraction and causal relationships.
5 BAC_5: flexible, precise discussion of complex or hypothetical implications.
6 BAC_6: nuanced synthesis, subtle distinctions, and well-controlled complex reasoning.
END SIX-BAND SPEAKING LADDER"""

SAFETY_CONSTRAINTS = """Reject any question that assumes overseas travel, family structure,
economic resources, device ownership, specialist knowledge, politics, religion, or regional
stereotypes. The prompt must be neutral and answerable by every Vietnamese high-school student."""

TOPICS = [
    ("School clubs", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Healthy routines", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Films and stories", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Games and technology", "TECH_GAMING", "OUT_OF_CURRICULUM"),
    ("Places in my town", "TRAVEL_PLACES", "IN_GDPT2018"),
    ("Everyday science", "FUTURE_SCIENCE", "IN_GDPT2018"),
    ("Learning with friends", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Sports choices", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Music and performance", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Future inventions", "FUTURE_SCIENCE", "IN_GDPT2018"),
]

CRITERIA = [
    ("PRONUNCIATION", None),
    ("FLUENCY", None),
    ("GRAMMAR", "sv_agreement"),
    ("GRAMMAR", "tense_control"),
    ("GRAMMAR", "complex_clause_control"),
    ("GRAMMAR", "third_person_s_omission"),
    ("GRAMMAR", "article_use"),
    ("GRAMMAR", "word_form"),
    ("VOCABULARY", "limited_range"),
    ("VOCABULARY", "repetition"),
    ("VOCABULARY", "weak_collocation"),
    ("COHERENCE", "weak_progression"),
    ("COHERENCE", "limited_support"),
    ("COHERENCE", "connector_overuse"),
    ("COHERENCE", "topic_drift"),
]

ALLOWED_SUB_ATTRIBUTES: dict[str, frozenset[str | None]] = {
    "PRONUNCIATION": frozenset({None}),
    "FLUENCY": frozenset({None}),
    "GRAMMAR": frozenset(
        {
            "sv_agreement",
            "tense_control",
            "complex_clause_control",
            "third_person_s_omission",
            "article_use",
            "word_form",
        }
    ),
    "VOCABULARY": frozenset(
        {"limited_range", "repetition", "weak_collocation"}
    ),
    "COHERENCE": frozenset(
        {
            "weak_progression",
            "limited_support",
            "connector_overuse",
            "topic_drift",
        }
    ),
}

FILTER_REASON_CODES = frozenset(
    {
        "NOT_ENGLISH",
        "LENGTH_OUT_OF_RANGE",
        "MISSING_FIELD",
        "SUB_ATTRIBUTE_NOT_ALLOWED",
        "CRITERION_UNKNOWN",
        "DUPLICATE_COSINE",
    }
)
