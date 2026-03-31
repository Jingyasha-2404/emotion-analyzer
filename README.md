# Emotion-Detector

A Python project that detects emotions in text using the Watson NLP API endpoint, provides a Flask web interface, includes unit tests, and supports static code analysis.

## Project Structure

- `EmotionDetection/`: importable package with emotion detection logic.
- `server.py`: Flask web application.
- `templates/index.html`: web UI for emotion analysis.
- `test_emotion_detection.py`: unit tests.
- `requirements.txt`: dependencies.

## Setup

1. Install dependencies:

   ```powershell
   C:/Users/bhaba/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install -r requirements.txt
   ```

2. Validate package import:

   ```powershell
   C:/Users/bhaba/AppData/Local/Python/pythoncore-3.14-64/python.exe -c "import EmotionDetection; print('EmotionDetection package is valid')"
   ```

3. Run tests:

   ```powershell
   C:/Users/bhaba/AppData/Local/Python/pythoncore-3.14-64/python.exe -m unittest test_emotion_detection.py -v
   ```

4. Run Flask app:

   ```powershell
   C:/Users/bhaba/AppData/Local/Python/pythoncore-3.14-64/python.exe server.py
   ```

5. Run static analysis:

   ```powershell
   C:/Users/bhaba/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pylint server.py
   ```
