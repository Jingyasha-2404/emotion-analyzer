"""Flask application for serving emotion detection results."""

from __future__ import annotations

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Render the single-page web interface."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route() -> tuple[str, int]:
    """Run emotion detection for user input and format the response."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again.", 400

    analysis = emotion_detector(text_to_analyze)
    if analysis["dominant_emotion"] is None:
        return "Invalid text! Please try again.", 400

    message = (
        "For the given statement, the system response is "
        f"'anger': {analysis['anger']}, "
        f"'disgust': {analysis['disgust']}, "
        f"'fear': {analysis['fear']}, "
        f"'joy': {analysis['joy']} and "
        f"'sadness': {analysis['sadness']}. "
        f"The dominant emotion is {analysis['dominant_emotion']}."
    )
    return message, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
