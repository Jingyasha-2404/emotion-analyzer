"""Unit tests for the emotion detector."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from EmotionDetection import emotion_detector


def _mock_watson_post(*args, **kwargs):
    """Return predictable mocked payloads keyed by input text."""
    text = kwargs["json"]["raw_document"]["text"]
    emotion_map = {
        "I am glad this happened": {
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.02,
            "joy": 0.92,
            "sadness": 0.04,
        },
        "I am really mad about this": {
            "anger": 0.91,
            "disgust": 0.03,
            "fear": 0.02,
            "joy": 0.01,
            "sadness": 0.03,
        },
        "I feel disgusted just hearing about this": {
            "anger": 0.03,
            "disgust": 0.9,
            "fear": 0.02,
            "joy": 0.01,
            "sadness": 0.04,
        },
        "I am so sad about this": {
            "anger": 0.02,
            "disgust": 0.01,
            "fear": 0.03,
            "joy": 0.02,
            "sadness": 0.92,
        },
        "I am really afraid that this will happen": {
            "anger": 0.03,
            "disgust": 0.02,
            "fear": 0.9,
            "joy": 0.02,
            "sadness": 0.03,
        },
    }

    payload = {
        "emotionPredictions": [{"emotion": emotion_map[text]}],
    }
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    return response


class EmotionDetectorTests(unittest.TestCase):
    """Validate dominant emotion results for sample inputs."""

    def test_joy(self) -> None:
        """Verify joy-dominant text."""
        with patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_watson_post):
            response = emotion_detector("I am glad this happened")
        self.assertEqual(response["dominant_emotion"], "joy")

    def test_anger(self) -> None:
        """Verify anger-dominant text."""
        with patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_watson_post):
            response = emotion_detector("I am really mad about this")
        self.assertEqual(response["dominant_emotion"], "anger")

    def test_disgust(self) -> None:
        """Verify disgust-dominant text."""
        with patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_watson_post):
            response = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(response["dominant_emotion"], "disgust")

    def test_sadness(self) -> None:
        """Verify sadness-dominant text."""
        with patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_watson_post):
            response = emotion_detector("I am so sad about this")
        self.assertEqual(response["dominant_emotion"], "sadness")

    def test_fear(self) -> None:
        """Verify fear-dominant text."""
        with patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_watson_post):
            response = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(response["dominant_emotion"], "fear")

    def test_status_400_returns_none_result(self) -> None:
        """Verify invalid-text API status returns None values."""
        mocked_response = Mock(status_code=400)
        with patch("EmotionDetection.emotion_detection.requests.post", return_value=mocked_response):
            response = emotion_detector("")

        self.assertIsNone(response["dominant_emotion"])

    def test_status_400_non_empty_uses_fallback(self) -> None:
        """Verify HTTP 400 for non-empty text still produces a result."""
        mocked_response = Mock(status_code=400)
        with patch("EmotionDetection.emotion_detection.requests.post", return_value=mocked_response):
            response = emotion_detector("I am happy")

        self.assertEqual(response["dominant_emotion"], "joy")

    def test_request_failure_uses_local_fallback(self) -> None:
        """Verify non-empty text still returns a best-effort emotion result."""
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            side_effect=requests.RequestException,
        ):
            response = emotion_detector("I am very happy this is complete")

        self.assertEqual(response["dominant_emotion"], "joy")


if __name__ == "__main__":
    unittest.main()
