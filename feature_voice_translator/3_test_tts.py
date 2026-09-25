# 3_test_tts.py (Updated to select a specific voice)
import pyttsx3
import os

def find_hindi_voice(engine):
    """Finds and returns the ID of a Hindi voice, if available."""
    voices = engine.getProperty('voices')
    for voice in voices:
        # The language attribute or name can indicate it's a Hindi voice
        # This check might need adjustment based on your specific OS voice names
        if "hindi" in voice.name.lower() or "hi-in" in voice.id.lower():
            print(f"Found Hindi voice: {voice.name}")
            return voice.id
    return None

def speak_text(text_to_speak, engine, voice_id=None):
    """Speaks the text using a specific voice if provided."""
    print(f"Speaking: '{text_to_speak}'")
    
    if voice_id:
        engine.setProperty('voice', voice_id)
        
    engine.say(text_to_speak)
    engine.runAndWait()
    print("✅ Done.")

def read_and_speak():
    input_file = "translation.txt"

    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        print("Please run '2_test_translation.py' first to generate it.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        text_from_file = f.read()
        
    if text_from_file.strip():
        # Initialize the engine
        engine = pyttsx3.init()
        
        # Find and use the Hindi voice
        hindi_voice_id = find_hindi_voice(engine)
        
        if hindi_voice_id:
            speak_text(text_from_file, engine, voice_id=hindi_voice_id)
        else:
            print("\n--- WARNING ---")
            print("No Hindi voice found on your system.")
            print("The audio output might be silent or incorrect.")
            print("Please install the Hindi language pack with Speech features in Windows Settings.")
            # Fallback to default voice
            speak_text(text_from_file, engine)
    else:
        print("The translation file is empty. Nothing to speak.")

if __name__ == "__main__":
    read_and_speak()