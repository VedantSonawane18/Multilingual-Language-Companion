# main_app_fixed.py
# One-to-one real-time voice translator
# Whisper -> MyMemory translation -> gTTS
#
# IMPORTANT:
# GoogleTranslator has been removed because the unofficial Google endpoint
# was returning "You made too many requests to the server".
#
# This version also avoids the PortAudio Invalid sample rate error by checking
# the selected microphone's supported/default sample rate.

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

from deep_translator import MyMemoryTranslator
from gtts import gTTS
from playsound import playsound


MODEL_SIZE = "base"

VAD_AGGRESSIVENESS = 3
TARGET_VAD_FS = 16000
FRAME_DURATION_MS = 30

SPEECH_PAD_SECONDS = 0.30
SILENCE_TIMEOUT_SECONDS = 0.60

LANG_CODES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "japanese": "ja",
    "urdu": "ur",
}

LANG_GROUPS = {
    "hindi": ["hi", "ur"],
}

# MyMemoryTranslator does NOT reliably accept Whisper/ISO short codes such
# as "en" and "hi". It expects the language names used by its supported
# language table (for example "english" and "hindi").
MYMEMORY_LANG_NAMES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "ja": "japanese",
    "ur": "urdu",
}


def mymemory_lang(code):
    code = str(code or "").strip().lower()
    return MYMEMORY_LANG_NAMES.get(code, code)


def to_code(lang):
    if not lang:
        return None

    lang = str(lang).strip().lower()

    if lang in LANG_CODES.values():
        return lang

    return LANG_CODES.get(lang)


def unique_temp_wav():
    return os.path.join(
        tempfile.gettempdir(),
        f"capture_{uuid.uuid4().hex}.wav"
    )


def unique_temp_mp3():
    return os.path.join(
        tempfile.gettempdir(),
        f"tts_{uuid.uuid4().hex}.mp3"
    )


def resample_for_vad(audio, source_rate, target_rate=TARGET_VAD_FS):
    """Convert one microphone frame to 16 kHz for WebRTC VAD."""
    if source_rate == target_rate:
        return audio.astype(np.int16)

    if len(audio) < 2:
        return np.asarray(audio, dtype=np.int16)

    duration = len(audio) / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))

    source_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)

    resampled = np.interp(
        target_x,
        source_x,
        audio.astype(np.float32)
    )

    return np.clip(resampled, -32768, 32767).astype(np.int16)


class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("One-to-one Real-time Translator")
        self.geometry("920x700")

        self.my_language = ctk.StringVar(value="english")
        self.their_language = ctk.StringVar(value="hindi")
        self.selected_device = ctk.StringVar()

        self.is_listening_auto = False
        self.stop_listening_event = threading.Event()

        self.whisper_model = None

        self.ptt_active = False
        self.ptt_thread = None

        self.translation_lock = threading.Lock()
        self.last_translation = {}

        self.input_devices = self._get_input_devices()

        if self.input_devices:
            self.selected_device.set(
                self._choose_preferred_input_name()
            )
        else:
            self.selected_device.set("")

        self._create_widgets()

        threading.Thread(
            target=self._load_whisper_model,
            daemon=True
        ).start()

    # ---------------------------------------------------------
    # Audio device handling
    # ---------------------------------------------------------

    def _get_input_devices(self):
        devices = sd.query_devices()

        return {
            dev["name"]: i
            for i, dev in enumerate(devices)
            if dev["max_input_channels"] > 0
        }

    def _choose_preferred_input_name(self):
        names = list(self.input_devices.keys())

        for keyword in (
            "internal",
            "built-in",
            "built in",
            "microphone",
            "mic",
            "realtek",
            "default",
        ):
            for name in names:
                if keyword in name.lower():
                    return name

        return names[0]

    def _get_supported_sample_rate(self, device_id):
        """
        Prefer 16 kHz, otherwise use the microphone's default rate.
        This fixes:
            Invalid sample rate [PaErrorCode -9997]
        """
        try:
            sd.check_input_settings(
                device=device_id,
                channels=1,
                samplerate=TARGET_VAD_FS,
                dtype="int16",
            )
            return TARGET_VAD_FS
        except Exception:
            try:
                info = sd.query_devices(device_id, "input")
                return int(round(info["default_samplerate"]))
            except Exception:
                return 44100

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=12
        )

        main_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            main_frame,
            text="My Language:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=6)

        ctk.CTkOptionMenu(
            main_frame,
            variable=self.my_language,
            values=list(LANG_CODES.keys())
        ).grid(
            row=1,
            column=0,
            padx=6,
            sticky="ew"
        )

        ctk.CTkLabel(
            main_frame,
            text="Their Language:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=1, sticky="w", padx=6)

        ctk.CTkOptionMenu(
            main_frame,
            variable=self.their_language,
            values=list(LANG_CODES.keys())
        ).grid(
            row=1,
            column=1,
            padx=6,
            sticky="ew"
        )

        ctk.CTkLabel(
            main_frame,
            text="Select Microphone:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=6,
            pady=(8, 0)
        )

        dev_values = (
            list(self.input_devices.keys())
            if self.input_devices
            else ["No input devices"]
        )

        self.mic_dropdown = ctk.CTkOptionMenu(
            main_frame,
            variable=self.selected_device,
            values=dev_values
        )

        self.mic_dropdown.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=6,
            sticky="ew",
            pady=(0, 8)
        )

        btn_frame = ctk.CTkFrame(main_frame)

        btn_frame.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=6,
            pady=(6, 8)
        )

        btn_frame.grid_columnconfigure(
            (0, 1, 2, 3),
            weight=1
        )

        self.auto_button = ctk.CTkButton(
            btn_frame,
            text="Start Auto Mode",
            command=self._toggle_auto_mode
        )
        self.auto_button.grid(
            row=0,
            column=0,
            padx=4,
            sticky="ew"
        )

        self.stop_auto_button = ctk.CTkButton(
            btn_frame,
            text="Stop Auto Mode",
            command=self._stop_auto_mode
        )
        self.stop_auto_button.grid(
            row=0,
            column=1,
            padx=4,
            sticky="ew"
        )

        self.ptt_me = ctk.CTkButton(
            btn_frame,
            text="Hold to Talk — Me"
        )
        self.ptt_me.grid(
            row=0,
            column=2,
            padx=4,
            sticky="ew"
        )

        self.ptt_them = ctk.CTkButton(
            btn_frame,
            text="Hold to Talk — Them"
        )
        self.ptt_them.grid(
            row=0,
            column=3,
            padx=4,
            sticky="ew"
        )

        self.ptt_me.bind(
            "<ButtonPress-1>",
            lambda e: self._start_push_to_talk("me")
        )
        self.ptt_me.bind(
            "<ButtonRelease-1>",
            lambda e: self._stop_push_to_talk("me")
        )

        self.ptt_them.bind(
            "<ButtonPress-1>",
            lambda e: self._start_push_to_talk("them")
        )
        self.ptt_them.bind(
            "<ButtonRelease-1>",
            lambda e: self._stop_push_to_talk("them")
        )

        self.my_original_text = self._make_textbox(
            main_frame,
            row=5,
            column=0,
            placeholder="My original speech will appear here..."
        )

        self.their_translated_text = self._make_textbox(
            main_frame,
            row=5,
            column=1,
            placeholder="Translated for them will appear here..."
        )

        self.their_original_text = self._make_textbox(
            main_frame,
            row=6,
            column=0,
            placeholder="Their original speech will appear here..."
        )

        self.my_translated_text = self._make_textbox(
            main_frame,
            row=6,
            column=1,
            placeholder="Translated for me will appear here..."
        )

        self.status_label = ctk.CTkLabel(
            self,
            text="Status: Ready"
        )
        self.status_label.pack(
            side="bottom",
            fill="x",
            padx=8,
            pady=6
        )

    def _make_textbox(self, parent, row, column, placeholder):
        textbox = ctk.CTkTextbox(
            parent,
            wrap="word",
            height=120
        )

        textbox.grid(
            row=row,
            column=column,
            padx=(6, 3) if column == 0 else (3, 6),
            pady=6,
            sticky="nsew"
        )

        textbox.insert("1.0", placeholder)
        textbox.configure(state="disabled")

        return textbox

    # ---------------------------------------------------------
    # Whisper
    # ---------------------------------------------------------

    def _load_whisper_model(self):
        self._set_status("Loading Whisper model...")

        try:
            self.whisper_model = whisper.load_model(MODEL_SIZE)
            self._set_status("Whisper loaded. Ready.")
        except Exception as e:
            self._set_status(f"Failed to load Whisper: {e}")

            self.after(
                0,
                lambda: tkmb.showerror(
                    "Model Error",
                    f"Failed to load Whisper model:\n\n{e}"
                )
            )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    def _set_status(self, text):
        self.after(
            0,
            lambda: self.status_label.configure(
                text=f"Status: {text}"
            )
        )

    # ---------------------------------------------------------
    # Auto mode
    # ---------------------------------------------------------

    def _toggle_auto_mode(self):
        if self.is_listening_auto:
            self._stop_auto_mode()
        else:
            self._start_auto_mode()

    def _start_auto_mode(self):
        if not self.whisper_model:
            tkmb.showwarning(
                "Whisper not ready",
                "Please wait for Whisper to finish loading."
            )
            return

        if not self.input_devices:
            tkmb.showerror(
                "Microphone Error",
                "No input devices found."
            )
            return

        if self.my_language.get() == self.their_language.get():
            tkmb.showwarning(
                "Language Error",
                "My Language and Their Language cannot be the same."
            )
            return

        self.is_listening_auto = True
        self.stop_listening_event.clear()

        self.auto_button.configure(
            text="Auto Mode: Running"
        )

        self.mic_dropdown.configure(
            state="disabled"
        )

        threading.Thread(
            target=self._realtime_conversation_loop,
            daemon=True
        ).start()

        self._set_status("Listening (auto mode)...")

    def _stop_auto_mode(self):
        self.stop_listening_event.set()
        self.is_listening_auto = False

        self.auto_button.configure(
            text="Start Auto Mode"
        )

        self.mic_dropdown.configure(
            state="normal"
        )

        self._set_status("Stopped auto mode.")

    # ---------------------------------------------------------
    # Auto VAD loop
    # ---------------------------------------------------------

    def _realtime_conversation_loop(self):
        selected_name = self.selected_device.get()
        device_id = self.input_devices.get(selected_name)

        actual_fs = self._get_supported_sample_rate(device_id)

        actual_chunk = int(
            actual_fs * FRAME_DURATION_MS / 1000
        )

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        padding_frames = max(
            3,
            int(
                SPEECH_PAD_SECONDS *
                TARGET_VAD_FS /
                (TARGET_VAD_FS * FRAME_DURATION_MS / 1000)
            )
        )

        ring_buffer = collections.deque(
            maxlen=padding_frames
        )

        triggered = False
        voiced_frames = []
        last_speech_time = None

        try:
            self._set_status(
                f"Listening... microphone rate: {actual_fs} Hz"
            )

            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=actual_fs,
                dtype="int16",
                blocksize=actual_chunk
            ) as stream:

                while not self.stop_listening_event.is_set():

                    try:
                        data, overflow = stream.read(actual_chunk)

                        if overflow:
                            continue

                    except Exception as e:
                        print("Stream read error:", e)
                        time.sleep(0.05)
                        continue

                    frame = np.asarray(
                        data
                    ).reshape(-1).astype(np.int16)

                    vad_frame = resample_for_vad(
                        frame,
                        actual_fs
                    )

                    # WebRTC requires exactly 10/20/30 ms at 8/16/32/48 kHz.
                    expected_vad_samples = int(
                        TARGET_VAD_FS *
                        FRAME_DURATION_MS /
                        1000
                    )

                    if len(vad_frame) != expected_vad_samples:
                        continue

                    try:
                        is_speech = vad.is_speech(
                            vad_frame.tobytes(),
                            TARGET_VAD_FS
                        )
                    except Exception:
                        is_speech = False

                    now = time.time()

                    if is_speech:
                        last_speech_time = now

                    if not triggered:

                        ring_buffer.append(
                            (frame.copy(), is_speech)
                        )

                        voiced_count = sum(
                            1
                            for _, speech in ring_buffer
                            if speech
                        )

                        if (
                            voiced_count >
                            0.5 * ring_buffer.maxlen
                        ):
                            triggered = True

                            self._set_status(
                                "Speech detected..."
                            )

                            for saved_frame, _ in ring_buffer:
                                voiced_frames.append(
                                    saved_frame.copy()
                                )

                            ring_buffer.clear()

                    else:

                        voiced_frames.append(
                            frame.copy()
                        )

                        ring_buffer.append(
                            (frame.copy(), is_speech)
                        )

                        if (
                            last_speech_time
                            and
                            now - last_speech_time
                            > SILENCE_TIMEOUT_SECONDS
                        ):

                            audio_np = np.concatenate(
                                voiced_frames,
                                axis=0
                            )

                            if audio_np.size:
                                tmp_wav = unique_temp_wav()

                                try:
                                    sf.write(
                                        tmp_wav,
                                        audio_np,
                                        actual_fs,
                                        subtype="PCM_16"
                                    )

                                    threading.Thread(
                                        target=self._process_captured_wav,
                                        args=(tmp_wav,),
                                        daemon=True
                                    ).start()

                                except Exception as e:
                                    print(
                                        "Failed to write WAV:",
                                        e
                                    )

                            triggered = False
                            voiced_frames = []
                            ring_buffer.clear()
                            last_speech_time = None

        except Exception as e:
            print("Auto mode audio error:", e)

            self._set_status(
                f"Audio error: {e}"
            )

            self.after(
                0,
                self._stop_auto_mode
            )

    # ---------------------------------------------------------
    # Whisper processing
    # ---------------------------------------------------------

    def _process_captured_wav(
        self,
        wav_path,
        forced_speaker_code=None,
        ptt_direction=None
    ):

        self._set_status("Transcribing...")

        transcribed_text = ""

        try:

            if forced_speaker_code:
                result = self.whisper_model.transcribe(
                    wav_path,
                    language=forced_speaker_code,
                    fp16=False
                )
            else:
                result = self.whisper_model.transcribe(
                    wav_path,
                    fp16=False
                )

            detected_language = result.get(
                "language"
            )

            transcribed_text = (
                result.get("text", "")
                .strip()
            )

        except Exception as e:
            print(
                "Whisper transcribe failed:",
                e
            )

            detected_language = None

        finally:

            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

        if not transcribed_text:
            self._set_status("Listening...")
            return

        # Explicit PTT direction.
        if ptt_direction == "me->them":

            self._update_textbox(
                self.my_original_text,
                transcribed_text
            )

            source = to_code(
                self.my_language.get()
            )

            target = to_code(
                self.their_language.get()
            )

            translated = self._translate_text(
                transcribed_text,
                source,
                target
            )

            self._update_textbox(
                self.their_translated_text,
                translated or "[Translation failed]"
            )

            if translated:
                threading.Thread(
                    target=self._speak_and_cleanup,
                    args=(translated, target),
                    daemon=True
                ).start()

            self._set_status("Listening...")
            return

        if ptt_direction == "them->me":

            self._update_textbox(
                self.their_original_text,
                transcribed_text
            )

            source = to_code(
                self.their_language.get()
            )

            target = to_code(
                self.my_language.get()
            )

            translated = self._translate_text(
                transcribed_text,
                source,
                target
            )

            self._update_textbox(
                self.my_translated_text,
                translated or "[Translation failed]"
            )

            if translated:
                threading.Thread(
                    target=self._speak_and_cleanup,
                    args=(translated, target),
                    daemon=True
                ).start()

            self._set_status("Listening...")
            return

        # Auto direction using Whisper's detected language.
        final_code = detected_language

        my_lang = self.my_language.get()
        their_lang = self.their_language.get()

        my_codes = [
            to_code(x)
            for x in LANG_GROUPS.get(
                my_lang,
                [my_lang]
            )
        ]

        their_codes = [
            to_code(x)
            for x in LANG_GROUPS.get(
                their_lang,
                [their_lang]
            )
        ]

        if final_code in my_codes:

            source_box = self.my_original_text
            target_box = self.their_translated_text
            target_code = to_code(their_lang)

        elif final_code in their_codes:

            source_box = self.their_original_text
            target_box = self.my_translated_text
            target_code = to_code(my_lang)

        else:

            # If Whisper could not determine the language,
            # default to "their" -> "my".
            source_box = self.their_original_text
            target_box = self.my_translated_text
            target_code = to_code(my_lang)

        self._update_textbox(
            source_box,
            transcribed_text
        )

        translated = self._translate_text(
            transcribed_text,
            final_code or "en",
            target_code
        )

        self._update_textbox(
            target_box,
            translated or "[Translation failed]"
        )

        if translated:
            threading.Thread(
                target=self._speak_and_cleanup,
                args=(translated, target_code),
                daemon=True
            ).start()

        self._set_status("Listening...")

    # ---------------------------------------------------------
    # Translation
    # ---------------------------------------------------------

    def _translate_text(
        self,
        text,
        source_code,
        target_code
    ):

        if not text or not text.strip():
            return None

        if not target_code:
            return None

        source_code = (
            source_code
            if source_code and source_code != "auto"
            else "en"
        )

        target_code = str(target_code).lower()

        if source_code == target_code:
            return text

        cache_key = (
            text.strip(),
            source_code,
            target_code
        )

        with self.translation_lock:

            if cache_key in self.last_translation:
                return self.last_translation[
                    cache_key
                ]

        try:

            # MyMemory replaces GoogleTranslator.
            # This avoids the Google 5 requests/second
            # error seen in the previous implementation.
            translator = MyMemoryTranslator(
                source=mymemory_lang(source_code),
                target=mymemory_lang(target_code)
            )

            result = translator.translate(
                text.strip()
            )

            if not result:
                return None

            result = result.strip()

            with self.translation_lock:
                self.last_translation[
                    cache_key
                ] = result

            return result

        except Exception as e:

            print(
                f"Translate error "
                f"({source_code} -> {target_code}): {e}"
            )

            return None

    # ---------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------

    def _update_textbox(
        self,
        textbox_widget,
        text_content
    ):

        def update():
            textbox_widget.configure(
                state="normal"
            )

            textbox_widget.delete(
                "1.0",
                "end"
            )

            textbox_widget.insert(
                "1.0",
                text_content or ""
            )

            textbox_widget.configure(
                state="disabled"
            )

        self.after(0, update)

    # ---------------------------------------------------------
    # TTS
    # ---------------------------------------------------------

    def _speak_and_cleanup(
        self,
        text,
        lang_code
    ):

        if not text or not lang_code:
            return

        tmp_mp3 = unique_temp_mp3()

        try:

            tts = gTTS(
                text=text,
                lang=lang_code,
                slow=False
            )

            tts.save(tmp_mp3)
            playsound(tmp_mp3)

        except Exception as e:

            print(
                "TTS/playback error:",
                e
            )

        finally:

            try:
                if os.path.exists(tmp_mp3):
                    os.remove(tmp_mp3)
            except Exception:
                pass

    # ---------------------------------------------------------
    # Push to talk
    # ---------------------------------------------------------

    def _start_push_to_talk(self, mode):

        if self.ptt_active:
            return

        if not self.whisper_model:
            tkmb.showwarning(
                "Whisper not ready",
                "Please wait for Whisper to finish loading."
            )
            return

        if not self.input_devices:
            tkmb.showerror(
                "Mic error",
                "No input devices found."
            )
            return

        self.ptt_active = True

        self._set_status(
            "Push-to-talk: recording..."
        )

        self.ptt_thread = threading.Thread(
            target=self._ptt_capture_loop,
            args=(mode,),
            daemon=True
        )

        self.ptt_thread.start()

    def _stop_push_to_talk(self, mode):
        self.ptt_active = False

    def _ptt_capture_loop(self, mode):

        selected_name = self.selected_device.get()
        device_id = self.input_devices.get(
            selected_name
        )

        actual_fs = self._get_supported_sample_rate(
            device_id
        )

        actual_chunk = int(
            actual_fs *
            FRAME_DURATION_MS /
            1000
        )

        frames = []

        try:

            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=actual_fs,
                dtype="int16",
                blocksize=actual_chunk
            ) as stream:

                while self.ptt_active:

                    try:

                        data, overflow = stream.read(
                            actual_chunk
                        )

                        if overflow:
                            continue

                        frames.append(
                            np.asarray(data)
                            .reshape(-1)
                            .astype(np.int16)
                        )

                    except Exception as e:

                        print(
                            "PTT stream read error:",
                            e
                        )

                        time.sleep(0.05)

            if not frames:
                self._set_status(
                    "Listening..."
                )
                return

            audio_np = np.concatenate(
                frames
            ).astype(np.int16)

            tmp_wav = unique_temp_wav()

            sf.write(
                tmp_wav,
                audio_np,
                actual_fs,
                subtype="PCM_16"
            )

            if mode == "me":

                forced_code = to_code(
                    self.my_language.get()
                )

                direction = "me->them"

            else:

                forced_code = to_code(
                    self.their_language.get()
                )

                direction = "them->me"

            threading.Thread(
                target=self._process_captured_wav,
                args=(
                    tmp_wav,
                    forced_code,
                    direction
                ),
                daemon=True
            ).start()

        except Exception as e:

            print(
                "PTT capture error:",
                e
            )

            self._set_status(
                f"PTT audio error: {e}"
            )


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = TranslatorApp()
    app.mainloop()