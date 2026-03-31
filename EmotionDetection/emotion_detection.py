"""Core emotion detection module using the Watson NLP API endpoint."""

from __future__ import annotations

import re
from typing import Any

import requests

EMOTION_API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"

KEYWORD_WEIGHTS = {
    "anger": {"angry", "mad", "furious", "annoyed", "rage", "hate"},
    "disgust": {"disgust", "gross", "revolting", "nasty", "awful"},
    "fear": {"afraid", "fear", "scared", "terrified", "anxious", "worry"},
    "joy": {"happy", "glad", "joy", "great", "love", "awesome", "excited", "complete"},
    "sadness": {"sad", "upset", "depressed", "cry", "miserable", "unhappy"},
}


def _empty_result() -> dict[str, Any]:
    """Return the fallback result for invalid text or request failures."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _build_result(emotions: dict[str, float]) -> dict[str, Any]:
    """Build the standard result payload from emotion scores."""
    dominant_emotion = max(emotions, key=emotions.get)
    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }


def _local_fallback_result(text_to_analyze: str) -> dict[str, Any]:
    """Infer lightweight emotion scores from keywords when API is unavailable."""
    tokens = re.findall(r"[a-z']+", text_to_analyze.lower())
    scores = {emotion: 0.01 for emotion in KEYWORD_WEIGHTS}

    for token in tokens:
        for emotion, keywords in KEYWORD_WEIGHTS.items():
            if token in keywords:
                scores[emotion] += 1.0

    total = sum(scores.values())
    normalized = {
        emotion: round(value / total, 4)
        for emotion, value in scores.items()
    }
    return _build_result(normalized)


def emotion_detector(text_to_analyze: str) -> dict[str, Any]:
    """Analyze text and return emotion scores plus the dominant emotion."""
    if text_to_analyze is None or not text_to_analyze.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyze}}
    headers = {"grpc-metadata-mm-model-id": MODEL_ID}

    try:
        response = requests.post(
            EMOTION_API_URL,
            json=payload,
            headers=headers,
            timeout=10,
        )
    except requests.RequestException:
        return _local_fallback_result(text_to_analyze)

    if response.status_code == 400:
        if text_to_analyze.strip():
            return _local_fallback_result(text_to_analyze)
        return _empty_result()

    if response.status_code != 200:
        return _local_fallback_result(text_to_analyze)

    try:
        emotions = response.json()["emotionPredictions"][0]["emotion"]
    except (KeyError, IndexError, TypeError, ValueError):
        return _local_fallback_result(text_to_analyze)

    return _build_result(
        {
            "anger": float(emotions.get("anger", 0.0)),
            "disgust": float(emotions.get("disgust", 0.0)),
            "fear": float(emotions.get("fear", 0.0)),
            "joy": float(emotions.get("joy", 0.0)),
            "sadness": float(emotions.get("sadness", 0.0)),
        }
    )
