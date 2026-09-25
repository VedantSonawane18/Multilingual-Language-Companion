# main_app_fixed.py
# One-to-one real-time translator with Auto mode + Hold-to-talk (Push-to-talk)
# - Whisper for transcription
# - webrtcvad for phrase detection (auto mode)
# - langdetect + script heuristics for robust detection
# - deep_translator.GoogleTranslator for translation
# - gTTS + playsound for TTS playback

import os
import uuid
import tempfile
import threading
import time
import collections

import customtkinter as ctk
import tkinter.messagebox as tkmb
import sounddevice as sd
import numpy as np
import soundfile as sf
import webrtcvad
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound

# text-language detector
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# -----------------------
# Configuration
# -----------------------
MODEL_SIZE = "base"
VAD_AGGRESSIVENESS = 3
FS = 16000
FRAME_DURATION_MS = 30                     # must be 10,20 or 30 for webrtcvad
CHUNK_SIZE = int(FS * FRAME_DURATION_MS / 1000)
SPEECH_PAD_SECONDS = 0.3
SILENCE_TIMEOUT_SECONDS = 0.6

# Map friendly language names to codes used by translators & TTS
LANG_CODES = {
    "english": "en", "hindi": "hi", "marathi": "mr", "spanish": "es",
    "french": "fr", "german": "de", "japanese": "ja", "urdu": "ur"
}
# Optionally group languages (e.g. hindi group includes urdu code)
LANG_GROUPS = {"hindi": ["hi", "ur"]}

# -----------------------
# Helpers
# -----------------------
def to_code(lang):
    """Normalize language name or code to a short code (e.g. 'english' -> 'en', 'en' -> 'en')."""
    if not lang:
        return None
    lang = str(lang).strip().lower()
    if lang in set(LANG_CODES.values()):
        return lang
    if lang in LANG_CODES:
        return LANG_CODES.get(lang)
    for name, code in LANG_CODES.items():
        if name == lang:
            return code
    if len(lang) >= 2:
        return lang[:2]
    return None

def contains_devanagari(s: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in s)

def contains_arabic(s: str) -> bool:
    return any('\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' for ch in s)

def contains_cjk(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def unique_temp_wav():
    return os.path.join(tempfile.gettempdir(), f"capture_{uuid.uuid4().hex}.wav")

def unique_temp_mp3():
    return os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")

# -----------------------
# UI App
# -----------------------
class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("One-to-one Real-time Translator")
        self.geometry("920x700")
        self.grid_columnconfigure(0, weight=1)

        # State
        self.my_language = ctk.StringVar(value="english")
        self.their_language = ctk.StringVar(value="hindi")
        self.selected_device = ctk.StringVar()
        self.is_listening_auto = False
        self.stop_listening_event = threading.Event()
        self.whisper_model = None

        # Push-to-talk flags
        self.ptt_active = False
        self.ptt_thread = None

        # Build UI, list devices, load model
        self.input_devices = self._get_input_devices()  # mapping name->index
        if self.input_devices:
            # choose a sensible default: prefer internal mic if present
            default_name = self._choose_preferred_input_name()
            self.selected_device.set(default_name)
        else:
            self.selected_device.set("")

        self._create_widgets()
        threading.Thread(target=self._load_whisper_model, daemon=True).start()

    def _get_input_devices(self):
        devices = sd.query_devices()
        return {dev['name']: i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0}

    def _choose_preferred_input_name(self):
        # prefer names that have 'micro', 'internal', 'built-in', 'realtek'
        names = list(self.input_devices.keys())
        lowered = [n.lower() for n in names]
        for keyword in ("internal", "built-in", "built in", "microphone", "mic", "realtek", "stereo mix", "default"):
            for i, name in enumerate(lowered):
                if keyword in name:
                    return names[i]
        return names[0]  # fallback

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=12)
        main_frame.grid_columnconfigure((0,1), weight=1)

        # Language selection
        ctk.CTkLabel(main_frame, text="My Language:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w", padx=6)
        ctk.CTkOptionMenu(main_frame, variable=self.my_language, values=list(LANG_CODES.keys())).grid(row=1, column=0, padx=6, sticky="ew")

        ctk.CTkLabel(main_frame, text="Their Language:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=1, sticky="w", padx=6)
        ctk.CTkOptionMenu(main_frame, variable=self.their_language, values=list(LANG_CODES.keys())).grid(row=1, column=1, padx=6, sticky="ew")

        # Device selection
        ctk.CTkLabel(main_frame, text="Select Microphone:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(8,0))
        dev_values = list(self.input_devices.keys()) if self.input_devices else ["No input devices"]
        self.mic_dropdown = ctk.CTkOptionMenu(main_frame, variable=self.selected_device, values=dev_values)
        self.mic_dropdown.grid(row=3, column=0, columnspan=2, padx=6, sticky="ew", pady=(0,8))

        # Buttons: Auto mode + PTT (hold-to-talk)
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(6,8))
        btn_frame.grid_columnconfigure((0,1,2,3), weight=1)

        self.auto_button = ctk.CTkButton(btn_frame, text="Start Auto Mode", command=self._toggle_auto_mode)
        self.auto_button.grid(row=0, column=0, padx=4, sticky="ew")

        self.stop_auto_button = ctk.CTkButton(btn_frame, text="Stop Auto Mode", command=self._stop_auto_mode)
        self.stop_auto_button.grid(row=0, column=1, padx=4, sticky="ew")

        # Push-to-talk: hold-to-talk — bind press & release events
        self.ptt_me = ctk.CTkButton(btn_frame, text="Hold to Talk — Me")
        self.ptt_me.grid(row=0, column=2, padx=4, sticky="ew")
        self.ptt_them = ctk.CTkButton(btn_frame, text="Hold to Talk — Them")
        self.ptt_them.grid(row=0, column=3, padx=4, sticky="ew")

        # We need actual tkinter bindings for press/release:
        self.ptt_me.bind("<ButtonPress-1>", lambda e: self._start_push_to_talk(mode="me"))
        self.ptt_me.bind("<ButtonRelease-1>", lambda e: self._stop_push_to_talk(mode="me"))
        self.ptt_them.bind("<ButtonPress-1>", lambda e: self._start_push_to_talk(mode="them"))
        self.ptt_them.bind("<ButtonRelease-1>", lambda e: self._stop_push_to_talk(mode="them"))

        # Text boxes for conversation
        self.my_original_text = ctk.CTkTextbox(main_frame, wrap="word", height=120)
        self.my_original_text.grid(row=5, column=0, padx=(6,3), pady=6, sticky="nsew")
        self.my_original_text.insert("1.0", "My original speech will appear here...")
        self.my_original_text.configure(state="disabled")

        self.their_translated_text = ctk.CTkTextbox(main_frame, wrap="word", height=120)
        self.their_translated_text.grid(row=5, column=1, padx=(3,6), pady=6, sticky="nsew")
        self.their_translated_text.insert("1.0", "Translated for them will appear here...")
        self.their_translated_text.configure(state="disabled")

        self.their_original_text = ctk.CTkTextbox(main_frame, wrap="word", height=120)
        self.their_original_text.grid(row=6, column=0, padx=(6,3), pady=6, sticky="nsew")
        self.their_original_text.insert("1.0", "Their original speech will appear here...")
        self.their_original_text.configure(state="disabled")

        self.my_translated_text = ctk.CTkTextbox(main_frame, wrap="word", height=120)
        self.my_translated_text.grid(row=6, column=1, padx=(3,6), pady=6, sticky="nsew")
        self.my_translated_text.insert("1.0", "Translated for me will appear here...")
        self.my_translated_text.configure(state="disabled")

        self.status_label = ctk.CTkLabel(self, text="Status: Ready")
        self.status_label.pack(side="bottom", fill="x", padx=8, pady=6)

    # Load Whisper
    def _load_whisper_model(self):
        self._set_status("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model(MODEL_SIZE)
            self._set_status("Models loaded. Ready.")
        except Exception as e:
            self._set_status(f"Failed to load model: {e}")
            tkmb.showerror("Model error", f"Failed to load Whisper model: {e}")

    def _set_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=f"Status: {text}"))

    # Auto mode toggle
    def _toggle_auto_mode(self):
        if self.is_listening_auto:
            self._stop_auto_mode()
        else:
            self._start_auto_mode()

    def _start_auto_mode(self):
        # Validate device and languages
        if not self.input_devices:
            tkmb.showerror("Mic error", "No input devices found.")
            return
        if self.my_language.get() == self.their_language.get():
            tkmb.showwarning("Language error", "My Language and Their Language cannot be the same.")
            return

        self.is_listening_auto = True
        self.stop_listening_event.clear()
        self.auto_button.configure(text="Auto Mode: Running")
        self.mic_dropdown.configure(state="disabled")
        threading.Thread(target=self._realtime_conversation_loop, daemon=True).start()
        self._set_status("Listening (auto mode)...")

    def _stop_auto_mode(self):
        self.stop_listening_event.set()
        self.is_listening_auto = False
        self.auto_button.configure(text="Start Auto Mode")
        self.mic_dropdown.configure(state="normal")
        self._set_status("Stopped auto mode.")

    # VAD-based auto loop
    def _realtime_conversation_loop(self):
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        selected_device_name = self.selected_device.get()
        selected_device_id = self.input_devices.get(selected_device_name, None)

        padding_frames_count = max(3, int(SPEECH_PAD_SECONDS * FS / CHUNK_SIZE))
        ring_buffer = collections.deque(maxlen=padding_frames_count)
        triggered = False
        voiced_frames = []
        last_speech_time = None

        try:
            with sd.InputStream(device=selected_device_id, channels=1, samplerate=FS, dtype='int16', blocksize=CHUNK_SIZE) as stream:
                while not self.stop_listening_event.is_set():
                    try:
                        data, overflow = stream.read(CHUNK_SIZE)
                    except Exception as e:
                        print("Stream read error:", e)
                        time.sleep(0.01)
                        continue
                    if overflow:
                        continue

                    frame = np.copy(data).reshape(-1)
                    frame_bytes = frame.tobytes()
                    try:
                        is_speech = vad.is_speech(frame_bytes, FS)
                    except Exception:
                        is_speech = False

                    now = time.time()
                    if is_speech:
                        last_speech_time = now

                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        num_voiced = len([1 for f, s in ring_buffer if s])
                        if num_voiced > 0.5 * ring_buffer.maxlen:
                            self.after(0, self._set_status, "Speech detected...")
                            triggered = True
                            for f, s in ring_buffer:
                                voiced_frames.append(f.copy())
                            ring_buffer.clear()
                    else:
                        voiced_frames.append(frame.copy())
                        ring_buffer.append((frame.copy(), is_speech))
                        if last_speech_time and (now - last_speech_time) > SILENCE_TIMEOUT_SECONDS:
                            self.after(0, self._set_status, "End of phrase detected. Processing...")
                            try:
                                audio_np = np.concatenate([v.reshape(-1) for v in voiced_frames], axis=0)
                            except Exception:
                                audio_np = np.array([], dtype=np.int16)

                            if audio_np.size > 0:
                                tmp_wav = unique_temp_wav()
                                try:
                                    sf.write(tmp_wav, audio_np.astype('int16'), FS, subtype='PCM_16')
                                    threading.Thread(target=self._process_captured_wav, args=(tmp_wav,), daemon=True).start()
                                except Exception as e:
                                    print("Failed to write capture WAV:", e)
                                    if os.path.exists(tmp_wav):
                                        os.remove(tmp_wav)

                            triggered = False
                            voiced_frames = []
                            ring_buffer.clear()
                            last_speech_time = None
        except Exception as e:
            self.after(0, tkmb.showerror, "Audio Error", f"Audio stream error: {e}")
            self.stop_listening_event.set()
            self.after(0, self._stop_auto_mode)

    # Process captured wav (auto or ptt will call this)
    def _process_captured_wav(self, wav_path, forced_speaker_code=None, ptt_direction=None):
        # forced_speaker_code: if push-to-talk, we already know speaker language code (e.g. 'hi' if speaker is Hindi)
        # ptt_direction: "me->them" or "them->me" when using push-to-talk (so direction is deterministic)
        self.after(0, self._set_status, "Transcribing...")
        whisper_lang = None
        transcribed_text = ""
        try:
            if forced_speaker_code:
                res = self.whisper_model.transcribe(wav_path, language=forced_speaker_code, fp16=False)
            else:
                res = self.whisper_model.transcribe(wav_path, fp16=False)
            whisper_lang = res.get("language")
            transcribed_text = res.get("text", "").strip()
        except Exception as e:
            print("Whisper transcribe failed:", e)
            whisper_lang = None
            transcribed_text = ""
        finally:
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

        if not transcribed_text:
            self.after(0, self._set_status, "Listening...")
            return

        # If push-to-talk provides direction, follow it directly (no heavy detection)
        if ptt_direction:
            if ptt_direction == "me->them":
                # speaker is me
                self.after(0, self._update_textbox, self.my_original_text, transcribed_text)
                src_code = to_code(self.my_language.get())
                tgt_code = to_code(LANG_CODES.get(self.their_language.get()))
                translated = self._translate_text(transcribed_text, src_code, tgt_code)
                self.after(0, self._update_textbox, self.their_translated_text, translated or "[Translation failed]")
                threading.Thread(target=self._speak_and_cleanup, args=(translated, tgt_code), daemon=True).start()
                self.after(0, self._set_status, "Listening...")
                return
            elif ptt_direction == "them->me":
                # speaker is them
                self.after(0, self._update_textbox, self.their_original_text, transcribed_text)
                src_code = to_code(self.their_language.get())
                tgt_code = to_code(LANG_CODES.get(self.my_language.get()))
                translated = self._translate_text(transcribed_text, src_code, tgt_code)
                self.after(0, self._update_textbox, self.my_translated_text, translated or "[Translation failed]")
                threading.Thread(target=self._speak_and_cleanup, args=(translated, tgt_code), daemon=True).start()
                self.after(0, self._set_status, "Listening...")
                return

        # Normal auto-detection path:
        # Try text-based detection first, then whisper_lang, then script fallback
        text_lang = None
        try:
            if len(transcribed_text) >= 2:
                text_lang = detect(transcribed_text)
        except Exception:
            text_lang = None

        final_code = to_code(text_lang) or to_code(whisper_lang)

        if not final_code:
            if contains_devanagari(transcribed_text):
                final_code = 'hi'
            elif contains_arabic(transcribed_text):
                final_code = 'ur'
            elif contains_cjk(transcribed_text):
                final_code = 'ja'

        my_lang = self.my_language.get()
        their_lang = self.their_language.get()

        my_lang_codes = [to_code(c) for c in LANG_GROUPS.get(my_lang, [LANG_CODES.get(my_lang)])]
        my_lang_codes = [c for c in my_lang_codes if c]
        their_lang_codes = [to_code(c) for c in LANG_GROUPS.get(their_lang, [LANG_CODES.get(their_lang)])]
        their_lang_codes = [c for c in their_lang_codes if c]

        # Decide direction
        source_textbox = None
        target_textbox = None
        target_lang_code = None

        if final_code in my_lang_codes:
            # spoken in my language -> translate to their language
            source_textbox = self.my_original_text
            target_textbox = self.their_translated_text
            target_lang_code = LANG_CODES.get(their_lang)
            # update source textbox
            self.after(0, self._update_textbox, source_textbox, transcribed_text)
        elif final_code in their_lang_codes:
            # spoken in their language -> translate to my language
            source_textbox = self.their_original_text
            target_textbox = self.my_translated_text
            target_lang_code = LANG_CODES.get(my_lang)
            self.after(0, self._update_textbox, source_textbox, transcribed_text)
        else:
            # ambiguous - assume they spoke 'their' language (default)
            source_textbox = self.their_original_text
            target_textbox = self.my_translated_text
            target_lang_code = LANG_CODES.get(my_lang)
            self.after(0, self._update_textbox, source_textbox, transcribed_text)

        # Translate
        translated_text = None
        try:
            src = final_code if final_code else 'auto'
            translated_text = GoogleTranslator(source=src, target=target_lang_code).translate(transcribed_text)
        except Exception as e:
            print("Translation failed:", e)
            translated_text = None

        if translated_text is None:
            translated_text = "[Translation failed]"

        # write translation to UI and speak
        self.after(0, self._update_textbox, target_textbox, translated_text)
        threading.Thread(target=self._speak_and_cleanup, args=(translated_text, target_lang_code), daemon=True).start()
        self.after(0, self._set_status, "Listening...")

    def _translate_text(self, text, source_code, target_code):
        try:
            return GoogleTranslator(source=source_code or "auto", target=target_code).translate(text)
        except Exception as e:
            print("Translate error:", e)
            return None

    def _update_textbox(self, textbox_widget, text_content):
        textbox_widget.configure(state="normal")
        textbox_widget.delete("1.0", "end")
        textbox_widget.insert("1.0", text_content or "")
        textbox_widget.configure(state="disabled")

    def _speak_and_cleanup(self, text, lang_code):
        if not text or not lang_code:
            return
        tmp_mp3 = unique_temp_mp3()
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(tmp_mp3)
            playsound(tmp_mp3)
        except Exception as e:
            print("TTS/playback error:", e)
        finally:
            try:
                if os.path.exists(tmp_mp3):
                    os.remove(tmp_mp3)
            except Exception:
                pass

    # -------------------------
    # Push-to-talk logic (hold-to-talk)
    # -------------------------
    def _start_push_to_talk(self, mode):
        # mode: "me" or "them"
        if self.ptt_active:
            return
        if not self.input_devices:
            tkmb.showerror("Mic error", "No input devices found.")
            return

        self.ptt_active = True
        self._set_status("Push-to-talk: recording...")
        # Start thread to capture until release
        self.ptt_thread = threading.Thread(target=self._ptt_capture_loop, args=(mode,), daemon=True)
        self.ptt_thread.start()

    def _stop_push_to_talk(self, mode):
        # flag the capture thread to stop; it will process audio
        self.ptt_active = False
        # status updated after processing

    def _ptt_capture_loop(self, mode):
        # Collect frames while self.ptt_active True
        selected_device_id = self.input_devices.get(self.selected_device.get(), None)
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # not strictly needed for PTT, but kept for sanity
        frames = []
        try:
            with sd.InputStream(device=selected_device_id, channels=1, samplerate=FS, dtype='int16', blocksize=CHUNK_SIZE) as stream:
                # Wait until release
                while self.ptt_active:
                    try:
                        data, overflow = stream.read(CHUNK_SIZE)
                    except Exception as e:
                        print("PTT stream read error:", e)
                        time.sleep(0.01)
                        continue
                    if overflow:
                        continue
                    frames.append(np.copy(data).reshape(-1))
                # ptt stopped; process captured frames
                if frames:
                    audio_np = np.concatenate(frames, axis=0).astype(np.int16)
                    tmp_wav = unique_temp_wav()
                    try:
                        sf.write(tmp_wav, audio_np, FS, subtype='PCM_16')
                        # Determine forced speaker code and direction
                        if mode == "me":
                            forced_code = to_code(self.my_language.get())
                            direction = "me->them"
                        else:
                            forced_code = to_code(self.their_language.get())
                            direction = "them->me"

                        # process
                        threading.Thread(target=self._process_captured_wav, args=(tmp_wav, forced_code, direction), daemon=True).start()
                    except Exception as e:
                        print("PTT write WAV failed:", e)
                        try:
                            if os.path.exists(tmp_wav):
                                os.remove(tmp_wav)
                        except Exception:
                            pass
        except Exception as e:
            print("PTT capture error:", e)
        finally:
            # ensure flag reset
            self.ptt_active = False
            self.after(0, self._set_status, "Listening...")

# Run app
if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()