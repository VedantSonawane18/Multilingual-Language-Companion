# 2_test_translation.py (Updated to save its output)
from deep_translator import GoogleTranslator
import os

def test_translate():
    input_text_file = "transcription.txt"
    output_text_file = "translation.txt" # File to save the result

    # Check if the transcription file exists
    if not os.path.exists(input_text_file):
        print(f"Error: The file '{input_text_file}' was not found.")
        print("Please run the '1_test_stt.py' script first to generate it.")
        return

    # Read the text from the file
    with open(input_text_file, "r", encoding="utf-8") as f:
        text_to_translate = f.read()

    # If the file was empty, stop.
    if not text_to_translate.strip():
        print("The transcription file is empty. Nothing to translate.")
        return

    print(f"Original Text (from file): {text_to_translate}")

    # Translate from English to Hindi
    translated_text = GoogleTranslator(source='auto', target='hi').translate(text_to_translate)
    
    print(f"Translated Text (Hindi): {translated_text}")

    # --- NEW CODE ---
    # Save the translation to a new text file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(translated_text)
    print(f"✅ Translation saved to {output_text_file}")
    # --- END NEW CODE ---


if __name__ == "__main__":
    test_translate()