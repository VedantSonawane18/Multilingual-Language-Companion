# 1_test_stt.py (Updated to save output)
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

def list_audio_devices():
    """Lists available audio input devices."""
    print("Available audio input devices:")
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']}")
            input_devices.append(i)
    return input_devices

def test_transcribe():
    fs = 44100
    seconds = 5
    output_text_file = "transcription.txt" # File to save the result

    input_devices = list_audio_devices()
    if not input_devices:
        print("No input devices found!")
        return
        
    try:
        device_id = int(input("Please enter the ID of the microphone you want to use: "))
        if device_id not in input_devices:
            print("Invalid ID. Please choose from the list.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    print(f"\n🔴 Recording for 5 seconds using device {device_id}...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16, device=device_id)
    sd.wait()
    print("✅ Recording complete.")

    temp_audio_file = "temp_recording.wav"
    write(temp_audio_file, fs, myrecording)

    print("\nLoading Whisper model (base)...")
    model = whisper.load_model("base")
    print("Model loaded. Transcribing...")

    result = model.transcribe(temp_audio_file, fp16=False)
    transcribed_text = result["text"]
    
    print("-" * 20)
    print(f"Transcription Result: {transcribed_text}")
    print("-" * 20)

    # --- NEW CODE ---
    # Save the transcription to a text file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(transcribed_text)
    print(f"✅ Transcription saved to {output_text_file}")
    # --- END NEW CODE ---
    
    # Clean up the temporary audio file
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

if __name__ == "__main__":
    test_transcribe()