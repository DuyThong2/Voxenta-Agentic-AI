import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mappers.assessment_response_adapter import normalize_criterion
from node.evalGraph.LanguageQualityEvalNode.language_quality_eval_node_config import (
    _to_criterion,
)
from node.evalGraph.LanguageQualityEvalNode.language_quality_eval_prompt import SYSTEM_PROMPT
from schemas.scoring import CriterionScore


class WeaknessDataPathTests(unittest.TestCase):

    def test_criterion_score_serializes_new_fields_as_camel_case(self):
        criterion = CriterionScore(
            weakness_labels=["tense_control"],
            evidence_spans=["I go yesterday"],
            recommendation_tag="past_tense",
            matched_band_code="BAC_3",
        )

        payload = criterion.model_dump(by_alias=True)

        self.assertEqual(payload["weaknessLabels"], ["tense_control"])
        self.assertEqual(payload["evidenceSpans"], ["I go yesterday"])
        self.assertEqual(payload["recommendationTag"], "past_tense")
        self.assertEqual(payload["matchedBandCode"], "BAC_3")

    @patch("mappers.assessment_response_adapter.logger.warning")
    def test_normalize_criterion_filters_taxonomy_evidence_and_band(self, warning):
        normalized = normalize_criterion(
            {
                "score": 72,
                "weakness_labels": ["tense_control", "invented_label"],
                "evidence_spans": ["x" * 250, "two", "three", "four", "five", "six"],
                "recommendation_tag": "past_tense",
                "matched_band_code": "BAC_99",
            },
            "grammar",
            default_source="llm",
            allowed_band_codes=["BAC_1", "BAC_2", "BAC_3"],
        )

        self.assertEqual(normalized["weakness_labels"], ["tense_control"])
        self.assertEqual(len(normalized["evidence_spans"]), 5)
        self.assertEqual(len(normalized["evidence_spans"][0]), 200)
        self.assertEqual(normalized["matched_band_code"], "")
        self.assertEqual(warning.call_count, 2)

    def test_pronunciation_and_fluency_never_accept_llm_weakness_labels(self):
        pronunciation = normalize_criterion(
            {"weakness_labels": ["phoneme_s"]},
            "pronunciation",
            default_source="azure",
        )
        fluency = normalize_criterion(
            {"weakness_labels": ["slow_rate"]},
            "fluency",
            default_source="azure",
        )

        self.assertEqual(pronunciation["weakness_labels"], [])
        self.assertEqual(fluency["weakness_labels"], [])

    def test_language_diagnostics_do_not_change_the_score(self):
        response = {
            "score": 72,
            "subscores": {"accuracy": 70},
            "weakness_labels": ["tense_control"],
            "evidence_spans": ["I go yesterday"],
            "recommendation_tag": "past_tense",
            "matched_band_code": "BAC_2",
        }
        speaking_input = SimpleNamespace(
            criteria_frameworks=[
                SimpleNamespace(
                    criterion_key="grammar",
                    bands=[
                        SimpleNamespace(code="BAC_1"),
                        SimpleNamespace(code="BAC_2"),
                        SimpleNamespace(code="BAC_3"),
                    ],
                )
            ]
        )

        criterion = _to_criterion(response, speaking_input, "grammar")

        self.assertEqual(criterion.score, 72)
        self.assertEqual(criterion.status, "scored")
        self.assertEqual(criterion.source, "llm")
        self.assertEqual(criterion.matched_band_code, "BAC_2")
        self.assertIn("INDEPENDENT of the score", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
