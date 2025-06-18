import mss
import mss.tools
from PIL import Image
import pytesseract
import numpy as np
import cv2
import google.generativeai as genai
import os
import pyautogui
import pandas as pd
from difflib import SequenceMatcher
import time

class KahootSolverLogic:
    def __init__(self, config):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
        
        # Configure Gemini API (API key is now handled by GUI)
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.generative_model = genai.GenerativeModel('gemini-1.5-flash')

    def take_screenshot(self):
        """
        Takes a screenshot of the specified monitor.
        Returns a PIL Image object.
        """
        with mss.mss() as sct:
            monitor_number = self.config.get("monitor_number", 1)
            monitor = sct.monitors[monitor_number]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
            return img

    def preprocess_image_for_ocr(self, image: Image.Image):
        """
        Preprocesses the image for better OCR results.
        Converts to grayscale, applies thresholding, and optionally resizes.
        """
        img_np = np.array(image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(img_thresh)

    def extract_text_from_image(self, image: Image.Image):
        """
        Extracts text and bounding box data from a PIL Image using Tesseract OCR.
        Returns a pandas DataFrame.
        """
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        data = data.dropna(subset=['text'])
        data = data[data.conf != -1]
        return data

    def parse_kahoot_text(self, ocr_data: pd.DataFrame):
        """
        Parses the OCR data (DataFrame) to separate the question and answers,
        along with their bounding box coordinates.
        """
        question_text = ""
        answers_with_coords = []
        ocr_data = ocr_data.sort_values(by=['top', 'left']).reset_index(drop=True)

        screen_height = ocr_data['height'].max() + ocr_data['top'].max()
        question_area_threshold = self.config.get("question_area_threshold", 0.3)
        question_area_threshold_y = screen_height * question_area_threshold

        question_lines = []
        answer_blocks = []
        current_answer_block = []
        
        for index, row in ocr_data.iterrows():
            text = str(row['text']).strip()
            if not text:
                continue

            x, y, w, h = row['left'], row['top'], row['width'], row['height']

            if y < question_area_threshold_y:
                question_lines.append(text)
            else:
                if not current_answer_block:
                    current_answer_block.append({'text': text, 'coords': (x, y, w, h)})
                else:
                    last_y = current_answer_block[-1]['coords'][1]
                    if abs(y - last_y) < h * 1.5:
                        current_answer_block.append({'text': text, 'coords': (x, y, w, h)})
                    else:
                        answer_blocks.append(current_answer_block)
                        current_answer_block = [{'text': text, 'coords': (x, y, w, h)}]
        
        if current_answer_block:
            answer_blocks.append(current_answer_block)

        question_text = " ".join(question_lines)

        for block in answer_blocks:
            full_text = " ".join([item['text'] for item in block])
            min_x = min(item['coords'][0] for item in block)
            min_y = min(item['coords'][1] for item in block)
            max_x = max(item['coords'][0] + item['coords'][2] for item in block)
            max_y = max(item['coords'][1] + item['coords'][3] for item in block)
            block_coords = (min_x, min_y, max_x - min_x, max_y - min_y)
            answers_with_coords.append({'text': full_text, 'coords': block_coords})

        print(f"Parsed Question: {question_text}")
        print(f"Parsed Answers with Coords: {answers_with_coords}")
        return question_text, answers_with_coords

    def find_answer_with_ai(self, question: str, answers_with_coords: list):
        """
        Uses Gemini AI to find the correct answer from the given question and answers.
        Returns the predicted answer text.
        """
        if not self.generative_model:
            print("AI model not initialized. Cannot find answer.")
            return "No Answer Found (AI Not Initialized)"

        answers_only_text = [ans['text'] for ans in answers_with_coords]

        if not question or not answers_only_text:
            print("Warning: Question or answers are empty. Cannot use AI.")
            return "No Answer Found (Insufficient Data)"

        prompt = f"Given the following multiple-choice question and options, identify the single correct answer. Respond with ONLY the text of the correct answer, nothing else.\n\nQuestion: {question}\nOptions:\n"
        for i, answer in enumerate(answers_only_text):
            prompt += f"{i+1}. {answer}\n"

        try:
            response = self.generative_model.generate_content(prompt)
            predicted_answer = response.text.strip()
            return predicted_answer
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "No Answer Found (AI Error)"

    def click_answer_on_screen(self, predicted_answer: str, answers_with_coords: list):
        """
        Clicks the predicted answer on the screen by finding its bounding box.
        Uses fuzzy matching to find the best match between predicted answer and OCR text.
        """
        if not predicted_answer or not answers_with_coords:
            print("Cannot click: Predicted answer or answer coordinates are missing.")
            return

        best_match_score = 0
        best_match_coords = None
        ocr_match_threshold = self.config.get("ocr_match_threshold", 0.7)

        for answer_data in answers_with_coords:
            ocr_text = answer_data['text']
            coords = answer_data['coords']

            matcher = SequenceMatcher(None, predicted_answer.lower(), ocr_text.lower())
            score = matcher.ratio()

            print(f"Comparing '{predicted_answer}' with '{ocr_text}' (Score: {score:.2f})")

            if score > ocr_match_threshold and score > best_match_score:
                best_match_score = score
                best_match_coords = coords

        if best_match_coords:
            x, y, w, h = best_match_coords
            click_x = x + w // 2
            click_y = y + h // 2

            print(f"Found best match with score {best_match_score:.2f}. Clicking at ({click_x}, {click_y})...")
            pyautogui.click(x=click_x, y=click_y)
            time.sleep(0.5)
        else:
            print(f"Could not find a good match for '{predicted_answer}' on screen.")

    def run(self):
        print("\n--- Activating Kahoot Solver ---")

        print("Taking screenshot...")
        screenshot = self.take_screenshot()

        print("Preprocessing image for OCR...")
        processed_image = self.preprocess_image_for_ocr(screenshot)

        print("Extracting text and coordinates from image...")
        ocr_data = self.extract_text_from_image(processed_image)
        print("\n--- Extracted OCR Data (first 10 rows) ---")
        print(ocr_data.head(10))
        print("------------------------------------------\n")

        question, answers_with_coords = self.parse_kahoot_text(ocr_data)

        print("Finding correct answer using AI...")
        correct_answer = self.find_answer_with_ai(question, answers_with_coords)
        print(f"Predicted correct answer: {correct_answer}")

        print(f"Attempting to click: {correct_answer}")
        self.click_answer_on_screen(correct_answer, answers_with_coords)

        print("--- Kahoot Solver Finished ---\n")