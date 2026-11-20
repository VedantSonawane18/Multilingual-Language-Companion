# conference_translator.py
# Conference / One-to-Many real-time translator
#
# Pipeline:
# Microphone -> WebRTC VAD -> Whisper -> MyMemory -> TTS
#
# GoogleTranslator has intentionally been removed.
# The previous implementation hit Google's unofficial endpoint and failed with:
# "You made too many requests to the server."
#
# Translation requests are now:
#   1. Cached
#   2. Sent sequentially rather than all at once
#   3. Rate-limited between requests
#
# This is much safer for the current demo and can later be replaced by
# Google Cloud Translation / Azure / DeepL with an API key.

import os
import uuid
import tempfile
import threading
import time
import collections
import datetime
import logging
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import customtkinter as ctk
from tkinter import filedialog, messagebox

import sounddevice as sd
import soundfile as sf
import numpy as np
import webrtcvad
import whisper

from deep_translator import MyMemoryTranslator

from gtts import gTTS
from playsound import playsound


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("translator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

class Config:
    TARGET_VAD_FS = 16000
    FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 3

    PRE_SPEECH_SECONDS = 0.5
    SILENCE_TIMEOUT_SECONDS = 1.0
    MAX_PHRASE_SECONDS = 45

    DEFAULT_MODEL = "base"
    CONFIDENCE_THRESHOLD = 0.7

    # Do NOT hammer public translation services.
    # 0.35 sec = < 3 requests/sec even when multiple languages
    # are requested.
    TRANSLATION_DELAY_SECONDS = 0.35

    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900

    PRIMARY_COLOR = "#1a1a2e"
    SECONDARY_COLOR = "#16213e"
    ACCENT_COLOR = "#0f3460"
    HIGHLIGHT_COLOR = "#e94560"
    TEXT_COLOR = "#ffffff"
    SUCCESS_COLOR = "#27ae60"
    WARNING_COLOR = "#f39c12"
    ERROR_COLOR = "#e74c3c"


# MyMemoryTranslator expects language names for reliable lookup.
# Whisper and the rest of this application use short ISO codes internally.
MYMEMORY_LANG_NAMES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "ja": "japanese",
    "ur": "urdu",
    "zh": "chinese simplified",
    "ar": "arabic",
    "ru": "russian",
    "pt": "portuguese",
    "it": "italian",
    "ko": "korean",
}


def mymemory_lang(code):
    code = str(code or "").strip().lower()
    return MYMEMORY_LANG_NAMES.get(code, code)


ENHANCED_LANG_CODES = {
    "english": {
        "code": "en",
        "tts": "en",
        "name": "English",
        "flag": "🇺🇸"
    },
    "hindi": {
        "code": "hi",
        "tts": "hi",
        "name": "हिन्दी",
        "flag": "🇮🇳"
    },
    "marathi": {
        "code": "mr",
        "tts": "mr",
        "name": "मराठी",
        "flag": "🇮🇳"
    },
    "spanish": {
        "code": "es",
        "tts": "es",
        "name": "Español",
        "flag": "🇪🇸"
    },
    "french": {
        "code": "fr",
        "tts": "fr",
        "name": "Français",
        "flag": "🇫🇷"
    },
    "german": {
        "code": "de",
        "tts": "de",
        "name": "Deutsch",
        "flag": "🇩🇪"
    },
    "japanese": {
        "code": "ja",
        "tts": "ja",
        "name": "日本語",
        "flag": "🇯🇵"
    },
    "urdu": {
        "code": "ur",
        "tts": "ur",
        "name": "اردو",
        "flag": "🇵🇰"
    },
    "chinese": {
        "code": "zh",
        "tts": "zh",
        "name": "中文",
        "flag": "🇨🇳"
    },
    "arabic": {
        "code": "ar",
        "tts": "ar",
        "name": "العربية",
        "flag": "🇸🇦"
    },
    "russian": {
        "code": "ru",
        "tts": "ru",
        "name": "Русский",
        "flag": "🇷🇺"
    },
    "portuguese": {
        "code": "pt",
        "tts": "pt",
        "name": "Português",
        "flag": "🇵🇹"
    },
    "italian": {
        "code": "it",
        "tts": "it",
        "name": "Italiano",
        "flag": "🇮🇹"
    },
    "korean": {
        "code": "ko",
        "tts": "ko",
        "name": "한국어",
        "flag": "🇰🇷"
    },
    "dutch": {
        "code": "nl",
        "tts": "nl",
        "name": "Nederlands",
        "flag": "🇳🇱"
    },
    "polish": {
        "code": "pl",
        "tts": "pl",
        "name": "Polski",
        "flag": "🇵🇱"
    },
    "turkish": {
        "code": "tr",
        "tts": "tr",
        "name": "Türkçe",
        "flag": "🇹🇷"
    },
    "swedish": {
        "code": "sv",
        "tts": "sv",
        "name": "Svenska",
        "flag": "🇸🇪"
    },
    "norwegian": {
        "code": "no",
        "tts": "no",
        "name": "Norsk",
        "flag": "🇳🇴"
    },
    "danish": {
        "code": "da",
        "tts": "da",
        "name": "Dansk",
        "flag": "🇩🇰"
    },
}


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    timestamp: datetime.datetime
    processing_time: float
    translation_engine: str


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def unique_temp_wav():
    return os.path.join(
        tempfile.gettempdir(),
        f"conference_{uuid.uuid4().hex}.wav"
    )


def unique_temp_mp3():
    return os.path.join(
        tempfile.gettempdir(),
        f"conference_tts_{uuid.uuid4().hex}.mp3"
    )


def resample_for_vad(audio, source_rate):
    if source_rate == Config.TARGET_VAD_FS:
        return audio.astype(np.int16)

    if len(audio) < 2:
        return audio.astype(np.int16)

    duration = len(audio) / float(source_rate)
    target_len = max(
        1,
        int(round(
            duration * Config.TARGET_VAD_FS
        ))
    )

    x_old = np.linspace(
        0,
        1,
        len(audio),
        endpoint=False
    )

    x_new = np.linspace(
        0,
        1,
        target_len,
        endpoint=False
    )

    result = np.interp(
        x_new,
        x_old,
        audio.astype(np.float32)
    )

    return np.clip(
        result,
        -32768,
        32767
    ).astype(np.int16)


# ---------------------------------------------------------
# Translation Engine
# ---------------------------------------------------------

class TranslationEngine:
    """
    Stable translation engine for the current demo.

    Uses MyMemory instead of the unofficial GoogleTranslator endpoint.

    Important:
    Do not translate the same sentence repeatedly.
    Cache is checked first.

    Requests are intentionally sequential so that conference mode does
    not fire Hindi + Spanish + French requests simultaneously.
    """

    def __init__(self):
        self.cache = {}
        self.translation_history = []

        self.lock = threading.Lock()
        self.request_lock = threading.Lock()

        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        with self.request_lock:

            elapsed = (
                time.time() -
                self.last_request_time
            )

            required = (
                Config.TRANSLATION_DELAY_SECONDS
            )

            if elapsed < required:
                time.sleep(
                    required - elapsed
                )

            self.last_request_time = time.time()

    def translate_with_confidence(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        engine: str = "mymemory"
    ) -> TranslationResult:

        start_time = time.time()

        source_lang = (
            source_lang or "en"
        ).lower()

        target_lang = (
            target_lang or "en"
        ).lower()

        text = (text or "").strip()

        cache_key = hashlib.sha256(
            f"{text}|{source_lang}|{target_lang}".encode(
                "utf-8"
            )
        ).hexdigest()

        with self.lock:

            cached = self.cache.get(
                cache_key
            )

            if cached:
                return TranslationResult(
                    original_text=cached.original_text,
                    translated_text=cached.translated_text,
                    source_lang=cached.source_lang,
                    target_lang=cached.target_lang,
                    confidence=cached.confidence,
                    timestamp=cached.timestamp,
                    processing_time=(
                        time.time() - start_time
                    ),
                    translation_engine="mymemory-cache"
                )

        if not text:

            return TranslationResult(
                original_text="",
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
                timestamp=datetime.datetime.now(),
                processing_time=0.0,
                translation_engine="mymemory"
            )

        # Same-language translation does not need an API call.
        if source_lang == target_lang:

            result = TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=1.0,
                timestamp=datetime.datetime.now(),
                processing_time=(
                    time.time() - start_time
                ),
                translation_engine="identity"
            )

            with self.lock:
                self.cache[cache_key] = result

            return result

        try:

            self._wait_for_rate_limit()

            translator = MyMemoryTranslator(
                source=mymemory_lang(source_lang),
                target=mymemory_lang(target_lang)
            )

            translated = translator.translate(
                text
            )

            if not translated:
                raise RuntimeError(
                    "Translation service returned an empty result."
                )

            translated = translated.strip()

            confidence = (
                self._calculate_translation_confidence(
                    text,
                    translated,
                    source_lang,
                    target_lang
                )
            )

            result = TranslationResult(
                original_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=confidence,
                timestamp=datetime.datetime.now(),
                processing_time=(
                    time.time() - start_time
                ),
                translation_engine="mymemory"
            )

            with self.lock:
                self.cache[cache_key] = result
                self.translation_history.append(
                    result
                )

            return result

        except Exception as e:

            logger.error(
                "Translation failed "
                f"({source_lang} -> {target_lang}): {e}"
            )

            return TranslationResult(
                original_text=text,
                translated_text=(
                    "[Translation failed: "
                    f"{str(e)}]"
                ),
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
                timestamp=datetime.datetime.now(),
                processing_time=(
                    time.time() - start_time
                ),
                translation_engine="mymemory"
            )

    def _calculate_translation_confidence(
        self,
        original,
        translated,
        source_lang,
        target_lang
    ):

        if not translated:
            return 0.0

        bad = (
            "translation failed",
            "unable to translate",
            "[translation failed"
        )

        if any(
            x in translated.lower()
            for x in bad
        ):
            return 0.1

        ratio = (
            len(translated) /
            max(len(original), 1)
        )

        if ratio < 0.05 or ratio > 15:
            return 0.3

        confidence = 0.85

        difficult_pairs = {
            ("en", "zh"),
            ("en", "ja"),
            ("en", "ar"),
            ("hi", "zh")
        }

        if (
            (source_lang, target_lang)
            in difficult_pairs
            or
            (target_lang, source_lang)
            in difficult_pairs
        ):
            confidence *= 0.9

        return min(
            confidence,
            1.0
        )


# ---------------------------------------------------------
# Conference Application
# ---------------------------------------------------------

class ConferenceTranslatorApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "Advanced Conference Translator Pro"
        )

        self.geometry(
            f"{Config.WINDOW_WIDTH}x"
            f"{Config.WINDOW_HEIGHT}"
        )

        self.minsize(
            1100,
            750
        )

        self.whisper_model = None

        self.translation_engine = (
            TranslationEngine()
        )

        self.is_listening = False

        self.stop_event = threading.Event()

        self.input_devices = (
            self._get_input_devices()
        )

        self.speaker_language = (
            ctk.StringVar(value="english")
        )

        self.audience_langs = (
            ctk.StringVar(
                value="hindi, spanish, french"
            )
        )

        self.whisper_model_var = (
            ctk.StringVar(
                value=Config.DEFAULT_MODEL
            )
        )

        # Keep this variable because the old UI used it.
        self.translation_engine_var = (
            ctk.StringVar(
                value="mymemory"
            )
        )

        self.session_data = {
            "translations": [],
            "session_start": datetime.datetime.now()
        }

        self._build_ui()

        threading.Thread(
            target=self._load_whisper_model,
            daemon=True
        ).start()

    # -----------------------------------------------------
    # Devices
    # -----------------------------------------------------

    def _get_input_devices(self):

        try:
            devices = sd.query_devices()

            return {
                dev["name"]: i
                for i, dev in enumerate(devices)
                if dev["max_input_channels"] > 0
            }

        except Exception as e:

            logger.error(
                f"Could not enumerate microphones: {e}"
            )

            return {}

    def _get_supported_sample_rate(
        self,
        device_id
    ):

        try:

            sd.check_input_settings(
                device=device_id,
                channels=1,
                samplerate=Config.TARGET_VAD_FS,
                dtype="int16"
            )

            return Config.TARGET_VAD_FS

        except Exception:

            try:

                info = sd.query_devices(
                    device_id,
                    "input"
                )

                return int(
                    round(
                        info["default_samplerate"]
                    )
                )

            except Exception:

                return 44100

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def _build_ui(self):

        header = ctk.CTkFrame(
            self,
            corner_radius=15
        )

        header.pack(
            fill="x",
            padx=15,
            pady=12
        )

        ctk.CTkLabel(
            header,
            text="🌐 Advanced Conference Translator Pro",
            font=ctk.CTkFont(
                size=25,
                weight="bold"
            ),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(
            pady=12
        )

        settings = ctk.CTkFrame(
            header
        )

        settings.pack(
            fill="x",
            padx=10,
            pady=(0, 10)
        )

        settings.grid_columnconfigure(
            (0, 1, 2, 3),
            weight=1
        )

        self._add_setting(
            settings,
            0,
            "Speaker Language:",
            self.speaker_language,
            list(
                ENHANCED_LANG_CODES.keys()
            )
        )

        self._add_setting(
            settings,
            1,
            "Audience Languages:",
            self.audience_langs,
            None
        )

        device_names = (
            list(self.input_devices.keys())
            if self.input_devices
            else ["No microphone"]
        )

        self.selected_device = (
            ctk.StringVar(
                value=device_names[0]
            )
        )

        self._add_setting(
            settings,
            2,
            "Microphone:",
            self.selected_device,
            device_names
        )

        self._add_setting(
            settings,
            3,
            "AI Model:",
            self.whisper_model_var,
            [
                "tiny",
                "base",
                "small",
                "medium",
                "large",
                "large-v2",
                "large-v3"
            ]
        )

        body = ctk.CTkFrame(
            self
        )

        body.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 12)
        )

        body.grid_columnconfigure(
            0,
            weight=1
        )

        body.grid_columnconfigure(
            1,
            weight=1
        )

        body.grid_rowconfigure(
            0,
            weight=1
        )

        left = ctk.CTkFrame(
            body,
            corner_radius=15
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6)
        )

        right = ctk.CTkFrame(
            body,
            corner_radius=15
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(6, 0)
        )

        self.start_btn = ctk.CTkButton(
            left,
            text="🎤 Start Translation",
            height=55,
            fg_color=Config.SUCCESS_COLOR,
            hover_color="#219150",
            command=self.start_listening
        )

        self.start_btn.pack(
            fill="x",
            padx=15,
            pady=(15, 6)
        )

        self.stop_btn = ctk.CTkButton(
            left,
            text="⏹ Stop Translation",
            height=55,
            fg_color=Config.ERROR_COLOR,
            hover_color="#c0392b",
            state="disabled",
            command=self.stop_listening
        )

        self.stop_btn.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.original_textbox = (
            ctk.CTkTextbox(
                left,
                height=220,
                wrap="word"
            )
        )

        self.original_textbox.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        self.original_textbox.insert(
            "1.0",
            "Speaker's original speech will appear here..."
        )

        self.confidence_indicator = (
            ctk.CTkLabel(
                left,
                text="Confidence: --",
                font=ctk.CTkFont(
                    size=14,
                    weight="bold"
                )
            )
        )

        self.confidence_indicator.pack(
            pady=(0, 10)
        )

        ctk.CTkLabel(
            right,
            text="🌍 Audience Translations",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(
            pady=12
        )

        self.translations_textbox = (
            ctk.CTkTextbox(
                right,
                wrap="word"
            )
        )

        self.translations_textbox.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        self.translations_textbox.insert(
            "1.0",
            "Translations will appear here..."
        )

        # Advanced settings
        advanced = ctk.CTkFrame(
            self
        )

        advanced.pack(
            fill="x",
            padx=15,
            pady=(0, 10)
        )

        ctk.CTkLabel(
            advanced,
            text="Translation Engine"
        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkLabel(
            advanced,
            text="MyMemory (rate-limited + cached)"
        ).pack(
            side="left",
            padx=5
        )

        ctk.CTkLabel(
            advanced,
            text="Confidence Threshold"
        ).pack(
            side="left",
            padx=(30, 5)
        )

        self.confidence_slider = (
            ctk.CTkSlider(
                advanced,
                from_=0.1,
                to=1.0,
                number_of_steps=9,
                command=self._on_confidence_change
            )
        )

        self.confidence_slider.set(
            Config.CONFIDENCE_THRESHOLD
        )

        self.confidence_slider.pack(
            side="left",
            padx=5
        )

        self.confidence_label = (
            ctk.CTkLabel(
                advanced,
                text=f"{Config.CONFIDENCE_THRESHOLD:.1f}"
            )
        )

        self.confidence_label.pack(
            side="left",
            padx=5
        )

        footer = ctk.CTkFrame(
            self
        )

        footer.pack(
            fill="x",
            padx=15,
            pady=(0, 10)
        )

        self.status_label = ctk.CTkLabel(
            footer,
            text="Status: Loading Whisper..."
        )

        self.status_label.pack(
            side="left",
            padx=10,
            pady=5
        )

        self.session_info = ctk.CTkLabel(
            footer,
            text="Ready"
        )

        self.session_info.pack(
            side="right",
            padx=10
        )

    def _add_setting(
        self,
        parent,
        column,
        label,
        variable,
        values
    ):

        frame = ctk.CTkFrame(
            parent
        )

        frame.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=5,
            pady=8
        )

        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            )
        ).pack(
            anchor="w",
            padx=8,
            pady=(6, 2)
        )

        if values:

            widget = ctk.CTkOptionMenu(
                frame,
                variable=variable,
                values=values
            )

        else:

            widget = ctk.CTkEntry(
                frame,
                textvariable=variable
            )

        widget.pack(
            fill="x",
            padx=8,
            pady=(0, 8)
        )

    # -----------------------------------------------------
    # Whisper
    # -----------------------------------------------------

    def _load_whisper_model(self):

        try:

            model = self.whisper_model_var.get()

            self._set_status(
                f"Loading Whisper {model}..."
            )

            self.whisper_model = (
                whisper.load_model(model)
            )

            self._set_status(
                "Whisper loaded. Ready."
            )

        except Exception as e:

            logger.error(
                f"Whisper loading failed: {e}"
            )

            self._set_status(
                f"Whisper error: {e}"
            )

            self.after(
                0,
                lambda: messagebox.showerror(
                    "Whisper Error",
                    str(e)
                )
            )

    # -----------------------------------------------------
    # UI helpers
    # -----------------------------------------------------

    def _set_status(self, text):

        self.after(
            0,
            lambda: self.status_label.configure(
                text=f"Status: {text}"
            )
        )

    def _on_confidence_change(self, value):

        Config.CONFIDENCE_THRESHOLD = float(
            value
        )

        self.confidence_label.configure(
            text=f"{float(value):.1f}"
        )

    # -----------------------------------------------------
    # Start / stop
    # -----------------------------------------------------

    def start_listening(self):

        if not self.whisper_model:

            messagebox.showwarning(
                "Whisper Not Ready",
                "Please wait for Whisper to finish loading."
            )

            return

        if not self.input_devices:

            messagebox.showerror(
                "Microphone Error",
                "No microphone was detected."
            )

            return

        audience = [
            x.strip().lower()
            for x in self.audience_langs.get().split(",")
            if x.strip()
        ]

        invalid = [
            x
            for x in audience
            if x not in ENHANCED_LANG_CODES
        ]

        if invalid:

            messagebox.showerror(
                "Invalid Audience Language",
                "Unsupported languages:\n\n"
                + ", ".join(invalid)
            )

            return

        self.audience_list = audience

        self.speaker_code = (
            ENHANCED_LANG_CODES[
                self.speaker_language.get()
            ]["code"]
        )

        self.session_data = {
            "translations": [],
            "session_start": datetime.datetime.now()
        }

        self.is_listening = True
        self.stop_event.clear()

        self.start_btn.configure(
            state="disabled"
        )

        self.stop_btn.configure(
            state="normal"
        )

        self._set_status(
            "🎤 Listening for speech..."
        )

        threading.Thread(
            target=self._vad_loop,
            daemon=True
        ).start()

    def stop_listening(self):

        self.stop_event.set()
        self.is_listening = False

        self.start_btn.configure(
            state="normal"
        )

        self.stop_btn.configure(
            state="disabled"
        )

        self._set_status(
            "⏹ Translation stopped"
        )

    # -----------------------------------------------------
    # VAD
    # -----------------------------------------------------

    def _vad_loop(self):

        device_id = self.input_devices.get(
            self.selected_device.get()
        )

        actual_fs = (
            self._get_supported_sample_rate(
                device_id
            )
        )

        actual_chunk = int(
            actual_fs *
            Config.FRAME_DURATION_MS /
            1000
        )

        vad = webrtcvad.Vad(
            Config.VAD_AGGRESSIVENESS
        )

        pre_buffer = collections.deque(
            maxlen=max(
                3,
                int(
                    Config.PRE_SPEECH_SECONDS /
                    (Config.FRAME_DURATION_MS / 1000)
                )
            )
        )

        voiced_frames = []

        triggered = False
        last_voice_time = None
        phrase_start_time = None

        try:

            logger.info(
                f"Microphone sample rate: {actual_fs} Hz"
            )

            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=actual_fs,
                dtype="int16",
                blocksize=actual_chunk
            ) as stream:

                while not self.stop_event.is_set():

                    data, overflow = (
                        stream.read(
                            actual_chunk
                        )
                    )

                    if overflow:
                        continue

                    frame = (
                        np.asarray(data)
                        .flatten()
                        .astype(np.int16)
                    )

                    vad_frame = (
                        resample_for_vad(
                            frame,
                            actual_fs
                        )
                    )

                    expected = int(
                        Config.TARGET_VAD_FS *
                        Config.FRAME_DURATION_MS /
                        1000
                    )

                    if len(vad_frame) != expected:
                        continue

                    try:

                        is_speech = vad.is_speech(
                            vad_frame.tobytes(),
                            Config.TARGET_VAD_FS
                        )

                    except Exception:

                        is_speech = False

                    now = time.time()

                    if not triggered:

                        pre_buffer.append(
                            (
                                frame.copy(),
                                is_speech
                            )
                        )

                        if len(pre_buffer) == pre_buffer.maxlen:

                            speech_ratio = (
                                sum(
                                    1
                                    for _, s
                                    in pre_buffer
                                    if s
                                )
                                /
                                len(pre_buffer)
                            )

                            if speech_ratio > 0.3:

                                triggered = True

                                phrase_start_time = now
                                last_voice_time = now

                                for saved, _ in pre_buffer:
                                    voiced_frames.append(
                                        saved
                                    )

                                pre_buffer.clear()

                                self._set_status(
                                    "🎤 Speech detected..."
                                )

                    else:

                        voiced_frames.append(
                            frame.copy()
                        )

                        if is_speech:
                            last_voice_time = now

                        phrase_duration = (
                            now - phrase_start_time
                        )

                        silence_duration = (
                            now - last_voice_time
                            if last_voice_time
                            else 0
                        )

                        should_process = (
                            phrase_duration
                            >= Config.MAX_PHRASE_SECONDS
                            or
                            silence_duration
                            >= Config.SILENCE_TIMEOUT_SECONDS
                        )

                        if should_process:

                            if voiced_frames:

                                audio = np.concatenate(
                                    voiced_frames
                                )

                                self._process_audio_phrase(
                                    audio,
                                    actual_fs
                                )

                            triggered = False
                            voiced_frames = []
                            pre_buffer.clear()
                            last_voice_time = None
                            phrase_start_time = None

        except Exception as e:

            logger.error(
                f"VAD loop failed: {e}"
            )

            self._set_status(
                f"❌ Audio error: {e}"
            )

            self.after(
                0,
                self.stop_listening
            )

    # -----------------------------------------------------
    # Audio processing
    # -----------------------------------------------------

    def _process_audio_phrase(
        self,
        audio_data,
        sample_rate
    ):

        self._set_status(
            "🔄 Transcribing..."
        )

        temp_wav = unique_temp_wav()

        try:

            sf.write(
                temp_wav,
                audio_data,
                sample_rate,
                subtype="PCM_16"
            )

            result = (
                self.whisper_model.transcribe(
                    temp_wav,
                    language=self.speaker_code,
                    fp16=False
                )
            )

            text = (
                result.get(
                    "text",
                    ""
                )
                .strip()
            )

            segments = result.get(
                "segments",
                []
            )

            if segments:

                logs = [
                    s.get(
                        "avg_logprob",
                        -1
                    )
                    for s in segments
                ]

                confidence = float(
                    np.mean(
                        [
                            np.exp(x)
                            for x in logs
                        ]
                    )
                )

                confidence = min(
                    max(
                        confidence,
                        0.0
                    ),
                    1.0
                )

            else:

                confidence = 0.7

        except Exception as e:

            logger.error(
                f"Transcription failed: {e}"
            )

            self._set_status(
                f"❌ Transcription error: {e}"
            )

            return

        finally:

            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception:
                pass

        if not text:

            self._set_status(
                "🎤 Listening..."
            )

            return

        self._update_original_text(
            text,
            confidence
        )

        self._translate_and_speak(
            text,
            confidence
        )

    # -----------------------------------------------------
    # One-to-many translation
    # -----------------------------------------------------

    def _translate_and_speak(
        self,
        original_text,
        transcription_confidence
    ):

        self._set_status(
            "🌍 Translating..."
        )

        display_lines = []

        # IMPORTANT:
        # Do NOT use ThreadPoolExecutor here.
        # The old version sent all languages simultaneously,
        # which made rate limiting much more likely.
        for lang_name in self.audience_list:

            lang_info = (
                ENHANCED_LANG_CODES.get(
                    lang_name
                )
            )

            if not lang_info:
                continue

            target_code = (
                lang_info["code"]
            )

            result = (
                self.translation_engine
                .translate_with_confidence(
                    original_text,
                    self.speaker_code,
                    target_code,
                    engine="mymemory"
                )
            )

            # Make the failure visible but do not crash
            # the remaining audience translations.
            if result.confidence == 0:

                display_lines.append(
                    f"{lang_info['flag']} "
                    f"{lang_info['name']}:\n"
                    f"{result.translated_text}"
                )

                continue

            display_lines.append(
                f"{lang_info['flag']} "
                f"{lang_info['name']}:\n"
                f"{result.translated_text}\n"
                f"Confidence: "
                f"{result.confidence:.1%}\n"
                f"Time: "
                f"{result.processing_time:.2f}s"
            )

            self.session_data[
                "translations"
            ].append(result)

            # TTS is intentionally sequential too.
            # This prevents several audio streams from
            # playing over each other.
            if (
                result.confidence
                >= Config.CONFIDENCE_THRESHOLD
            ):

                self._generate_and_play_tts(
                    result.translated_text,
                    lang_info["tts"]
                )

        self._update_translations(
            "\n\n".join(display_lines)
        )

        self._set_status(
            "🎤 Listening..."
        )

        self._update_session_info()

    # -----------------------------------------------------
    # UI translation display
    # -----------------------------------------------------

    def _update_original_text(
        self,
        text,
        confidence
    ):

        info = (
            ENHANCED_LANG_CODES[
                self.speaker_language.get()
            ]
        )

        display = (
            f"{info['flag']} "
            f"{info['name']}: "
            f"{text}"
        )

        def update():

            self.original_textbox.delete(
                "1.0",
                "end"
            )

            self.original_textbox.insert(
                "1.0",
                display
            )

            color = (
                Config.SUCCESS_COLOR
                if confidence > 0.8
                else Config.WARNING_COLOR
                if confidence > 0.5
                else Config.ERROR_COLOR
            )

            self.confidence_indicator.configure(
                text=f"Confidence: {confidence:.1%}",
                text_color=color
            )

        self.after(
            0,
            update
        )

    def _update_translations(
        self,
        text
    ):

        def update():

            self.translations_textbox.delete(
                "1.0",
                "end"
            )

            self.translations_textbox.insert(
                "1.0",
                text
            )

        self.after(
            0,
            update
        )

    def _update_session_info(self):

        count = len(
            self.session_data[
                "translations"
            ]
        )

        started = (
            self.session_data[
                "session_start"
            ]
        )

        duration = (
            datetime.datetime.now()
            - started
        )

        self.after(
            0,
            lambda: self.session_info.configure(
                text=(
                    f"Duration: "
                    f"{str(duration).split('.')[0]} "
                    f"| Translations: {count}"
                )
            )
        )

    # -----------------------------------------------------
    # TTS
    # -----------------------------------------------------

    def _generate_and_play_tts(
        self,
        text,
        lang
    ):

        if not text:
            return

        tmp_mp3 = unique_temp_mp3()

        try:

            tts = gTTS(
                text=text,
                lang=lang,
                slow=False
            )

            tts.save(
                tmp_mp3
            )

            playsound(
                tmp_mp3
            )

        except Exception as e:

            logger.warning(
                f"TTS failed: {e}"
            )

        finally:

            try:
                if os.path.exists(
                    tmp_mp3
                ):
                    os.remove(
                        tmp_mp3
                    )
            except Exception:
                pass

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    def export_json(self):

        if not self.session_data[
            "translations"
        ]:

            messagebox.showwarning(
                "No Data",
                "No translations to export."
            )

            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[
                (
                    "JSON files",
                    "*.json"
                )
            ]
        )

        if not path:
            return

        data = {
            "session_start": (
                self.session_data[
                    "session_start"
                ].isoformat()
            ),
            "translations": []
        }

        for item in self.session_data[
            "translations"
        ]:

            data[
                "translations"
            ].append({
                "original_text":
                    item.original_text,
                "translated_text":
                    item.translated_text,
                "source_language":
                    item.source_lang,
                "target_language":
                    item.target_lang,
                "confidence":
                    item.confidence,
                "timestamp":
                    item.timestamp.isoformat(),
                "processing_time":
                    item.processing_time,
                "translation_engine":
                    item.translation_engine
            })

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        messagebox.showinfo(
            "Export Complete",
            f"Saved to:\n{path}"
        )

    def on_closing(self):

        if self.is_listening:

            self.stop_listening()

        self.destroy()


if __name__ == "__main__":

    ctk.set_appearance_mode(
        "dark"
    )

    ctk.set_default_color_theme(
        "blue"
    )

    app = ConferenceTranslatorApp()

    app.protocol(
        "WM_DELETE_WINDOW",
        app.on_closing
    )

    app.mainloop()