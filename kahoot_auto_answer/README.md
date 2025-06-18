# Kahoot Auto Answer

This project provides an automated solver for Kahoot quizzes using OCR, AI (Google Gemini API), and screen interaction. It includes both a terminal version and a GUI version (currently in development).

## Features

- Captures screenshots of the Kahoot quiz.
- Extracts question and answer text using OCR.
- Uses Google Gemini AI to predict the correct answer.
- Automatically clicks the predicted answer on the screen.(*doesnt work yet*)
- Terminal version for command-line usage.
- GUI interface (in progress) to input API key, start/stop the solver, and view real-time logs.

## Requirements

- Python 3.7 or higher
- Tesseract OCR installed and configured
- Google Gemini API key

## Installation

1. Clone the repository.

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:

- Windows: Download from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) and install.
- Make sure to configure the `tesseract_cmd` path in `main.py` config.

## Usage

### Terminal Version

Run the solver directly from the command line:

```bash
python kahoot_auto_answer/main.py
```

Make sure to set the `GEMINI_API_KEY` environment variable before running.

### GUI Version (In Progress)

Run the GUI application:

```bash
python kahoot_auto_answer/gui.py
```

Enter your Gemini API key in the GUI, then click "Start Auto Answer" to begin. Use the "Stop Auto Answer" button to stop the process.

## Notes

- Ensure the Kahoot quiz is visible on the screen for accurate OCR.
- The AI model requires a valid Gemini API key.
- The project uses `pyautogui` for screen interaction; ensure permissions are granted.
- Please understand that the ai can make mistakes as well as misinterperating other words, this is entirely **experimental** and is only done for fun, however, i might improve this software, but i hate python, so time will tell

## License

This project is licensed under the MIT License.
