import math
import threading
import unittest
from unittest import mock

import utils.confidence_utils as confidence_utils
from infra.voice_live_client import VoiceLiveClient
from mappers.exam_event_builder import build_signals
from node.state_models.speaking_input import SpeakingInput
from utils.confidence_utils import (
    compute_alignment_confidence,
    compute_reference_confidence,
    quality_from_snr,
    quality_from_speech_ratio,
    run_consensus_judgment,
)
from utils.speech_client import compute_cross_asr_agreement, normalize_for_wer


class _Logprob:
    def __init__(self, value: float) -> None:
        self.logprob = value


class ConfidenceLayerTests(unittest.TestCase):
    def test_asr_log_uses_global_and_bottom_twenty_percent(self) -> None:
        logprobs = [_Logprob(math.log(value)) for value in (0.9, 0.9, 0.9, 0.9, 0.1)]

        confidence = VoiceLiveClient._confidence_from_logprobs(logprobs)

        expected = math.sqrt((0.9**4 * 0.1) ** (1 / 5) * 0.1)
        self.assertAlmostEqual(confidence, expected)

    def test_vietnamese_normalization_is_diacritic_insensitive(self) -> None:
        self.assertEqual(normalize_for_wer("Tôi ăn [VI: bánh mì]."), "toi an banh mi")
        self.assertEqual(
            compute_cross_asr_agreement("Tôi ăn bánh mì", "toi an banh mi"),
            1.0,
        )

    def test_audio_quality_clips_to_unit_interval(self) -> None:
        self.assertEqual(quality_from_snr(5.0), 0.0)
        self.assertEqual(quality_from_snr(7.5), 0.5)
        self.assertEqual(quality_from_snr(12.0), 1.0)
        self.assertEqual(quality_from_speech_ratio(0.8), 0.0)
        self.assertAlmostEqual(quality_from_speech_ratio(0.7), 0.5)
        self.assertEqual(quality_from_speech_ratio(0.5), 1.0)

    def test_reference_confidence_selects_medoid_and_penalizes_drift(self) -> None:
        reference, confidence, stability, drift = compute_reference_confidence(
            "I like green tea",
            (
                "I like green tea",
                "I really like green tea",
                "I like green tea",
            ),
        )

        self.assertEqual(reference, "I like green tea")
        self.assertGreater(stability, 0.8)
        self.assertEqual(drift, 0.0)
        self.assertAlmostEqual(confidence, stability)

    def test_alignment_confidence_uses_miscues_coverage_and_timing(self) -> None:
        segments = [{
            "NBest": [{
                "Words": [
                    {"Word": "one", "Duration": 10, "PronunciationAssessment": {"ErrorType": "None"}},
                    {"Word": "two", "Duration": 10, "PronunciationAssessment": {"ErrorType": "Omission"}},
                    {"Word": "extra", "Duration": 10, "PronunciationAssessment": {"ErrorType": "Insertion"}},
                    {"Word": "three", "Duration": 10, "PronunciationAssessment": {"ErrorType": "None"}},
                ]
            }]
        }]

        confidence = compute_alignment_confidence(segments, "one two three")

        self.assertAlmostEqual(confidence.composite, 1 / 3)

    def test_llm_consensus_normalizes_delta_from_100_point_scale(self) -> None:
        responses = [
            {"score": 70, "note": "The word student is used accurately."},
            {"score": 80, "note": "The word student supports the answer."},
            {"score": 90, "note": "The word student is clear."},
        ]
        lock = threading.Lock()

        def fake_call(_system: str, _user: str):
            with lock:
                return responses.pop()

        providers = ((fake_call, fake_call),) * 3
        with mock.patch.object(confidence_utils, "_CONSENSUS_PROVIDERS", providers):
            judgment = run_consensus_judgment(
                "system",
                'Transcript: "I am a student"',
                "I am a student",
            )

        self.assertEqual(judgment.response["score"], 80)
        self.assertEqual(judgment.score_delta_on_ten_point_scale, 2.0)
        self.assertEqual(judgment.confidence, 0.0)

    def test_event_builder_uses_nolog_branch_and_pf_is_min_not_gated(self) -> None:
        speaking_input = SpeakingInput(
            audio_path="unused.wav",
            realtime_transcript="I am a student",
            realtime_transcript_confidence=None,
        )
        result = {
            "speaking_input": speaking_input,
            "cross_asr_agreement": 0.95,
            "answer_length_metrics": {
                "word_count": 4,
                "sentence_count": 1,
                "expected_min_words": 4,
                "q_snr": 0.75,
                "q_speech": 1.0,
                "clipping_ratio": 0.002,
                "code_switching_ratio": 0.0,
            },
            "metadata": {
                "c_ref": 0.9,
                "c_align": 0.85,
                "grammar_confidence": 0.8,
                "vocabulary_confidence": 0.7,
                "coherence_confidence": 0.6,
            },
        }

        signals = build_signals(result)

        self.assertIsNone(signals.confidence_case.c_asr_log)
        self.assertEqual(signals.confidence_case.cross_asr_agreement, 0.95)
        self.assertEqual(signals.confidence_case.q_snr, 0.75)
        # c_pf_branch = min(c_ref, c_align) = min(0.9, 0.85) = 0.85 -- KHÔNG còn bị ép về 0 dù
        # asr_common (min 0.95/0.75/1.0 = 0.75) < 0.80. Việc route ASR do ConfidenceReviewCalculator
        # xử lý riêng, c_pf_branch giữ nguyên giá trị thật (khớp giả mã cuối research).
        self.assertEqual(signals.confidence_case.c_pf_branch, 0.85)
        self.assertEqual(signals.confidence_case.c_discourse, 0.6)
        # audio_quality: không có asr_confidence_avg/silence_ratio (mai-transcribe-1) -> fallback
        # min(q_snr, q_speech) = min(0.75, 1.0) = 0.75, thay vì null (UI hiện "-").
        self.assertEqual(signals.audio_quality, 0.75)

    def test_event_builder_aggregates_worst_turn_and_max_clipping(self) -> None:
        def turn(confidence: float, clipping: float):
            speaking_input = SpeakingInput(
                audio_path="unused.wav",
                realtime_transcript="answer",
                realtime_transcript_confidence=confidence,
            )
            return {
                "speaking_input": speaking_input,
                "answer_length_metrics": {
                    "word_count": 1,
                    "sentence_count": 1,
                    "expected_min_words": 1,
                    "clipping_ratio": clipping,
                },
                "metadata": {"c_ref": 0.9, "c_align": 0.95},
            }

        aggregate = {
            "speaking_input": turn(0.9, 0.001)["speaking_input"],
            "answer_length_metrics": {
                "word_count": 2,
                "sentence_count": 1,
                "expected_min_words": 2,
            },
            "metadata": {
                "grammar_confidence": 0.9,
                "vocabulary_confidence": 0.8,
                "coherence_confidence": 0.7,
            },
        }

        signals = build_signals(
            aggregate,
            duration_seconds=10,
            turn_results=[turn(0.9, 0.001), turn(0.7, 0.02)],
        )

        self.assertEqual(signals.confidence_case.c_asr_log, 0.7)
        self.assertEqual(signals.confidence_case.clipping_ratio, 0.02)
        self.assertEqual(signals.confidence_case.c_ref, 0.9)
        self.assertEqual(signals.confidence_case.c_discourse, 0.7)


if __name__ == "__main__":
    unittest.main()
