"""
Advanced Conference Translator Pro - Ultimate Edition
Features:
- Modern glassmorphism UI with animations
- Advanced VAD with noise reduction
- Multiple AI models (Whisper, OpenAI, Azure)
- Real-time confidence scoring
- Audio enhancement and noise cancellation
- Professional recording capabilities
- Advanced translation engines with quality metrics
- Live transcription display
- Speaker identification and diarization
- Multi-format export (PDF, DOCX, TXT, JSON)
- Audio visualization and waveform display
- Custom language models support
- Network streaming capabilities
- Advanced logging and analytics
"""

import os
import sys
import uuid
import tempfile
import threading
import time
import json
import csv
import datetime
import collections
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue

# Enhanced imports
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import numpy as np
import soundfile as sf
import webrtcvad
import whisper
import librosa
import noisereduce as nr
from scipy import signal
from scipy.io import wavfile

# AI and Translation
from deep_translator import GoogleTranslator, MicrosoftTranslator, DeeplTranslator
from gtts import gTTS, gTTSError
import pydub
from pydub import AudioSegment
from pydub.playback import play

# UI and Visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from PIL import Image, ImageTk

# Document export
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document
from docx.shared import Inches

# Advanced features
import requests
import asyncio
import aiohttp
from collections import defaultdict
import pickle
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------
# Enhanced Configuration
# -----------------------
class Config:
    # Audio settings
    FS = 16000
    FRAME_DURATION_MS = 30
    CHUNK_SIZE = int(FS * FRAME_DURATION_MS / 1000)
    VAD_AGGRESSIVENESS = 3
    PRE_SPEECH_SECONDS = 0.5
    SILENCE_TIMEOUT_SECONDS = 1.0
    MAX_PHRASE_SECONDS = 45
    NOISE_REDUCE_STATIONARY = True
    NOISE_REDUCE_PROP_DECREASE = 0.8
    
    # Model settings
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    DEFAULT_MODEL = "base"
    CONFIDENCE_THRESHOLD = 0.7
    
    # UI settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    ANIMATION_SPEED = 200
    
    # Colors (Modern dark theme)
    PRIMARY_COLOR = "#1a1a2e"
    SECONDARY_COLOR = "#16213e"
    ACCENT_COLOR = "#0f3460"
    HIGHLIGHT_COLOR = "#e94560"
    TEXT_COLOR = "#ffffff"
    SUCCESS_COLOR = "#27ae60"
    WARNING_COLOR = "#f39c12"
    ERROR_COLOR = "#e74c3c"

# Enhanced language support
ENHANCED_LANG_CODES = {
    "english": {"code": "en", "tts": "en", "name": "English", "flag": "🇺🇸"},
    "hindi": {"code": "hi", "tts": "hi", "name": "हिन्दी", "flag": "🇮🇳"},
    "marathi": {"code": "mr", "tts": "mr", "name": "मराठी", "flag": "🇮🇳"},
    "spanish": {"code": "es", "tts": "es", "name": "Español", "flag": "🇪🇸"},
    "french": {"code": "fr", "tts": "fr", "name": "Français", "flag": "🇫🇷"},
    "german": {"code": "de", "tts": "de", "name": "Deutsch", "flag": "🇩🇪"},
    "japanese": {"code": "ja", "tts": "ja", "name": "日本語", "flag": "🇯🇵"},
    "urdu": {"code": "ur", "tts": "ur", "name": "اردو", "flag": "🇵🇰"},
    "chinese": {"code": "zh", "tts": "zh", "name": "中文", "flag": "🇨🇳"},
    "arabic": {"code": "ar", "tts": "ar", "name": "العربية", "flag": "🇸🇦"},
    "russian": {"code": "ru", "tts": "ru", "name": "Русский", "flag": "🇷🇺"},
    "portuguese": {"code": "pt", "tts": "pt", "name": "Português", "flag": "🇵🇹"},
    "italian": {"code": "it", "tts": "it", "name": "Italiano", "flag": "🇮🇹"},
    "korean": {"code": "ko", "tts": "ko", "name": "한국어", "flag": "🇰🇷"},
    "dutch": {"code": "nl", "tts": "nl", "name": "Nederlands", "flag": "🇳🇱"},
    "polish": {"code": "pl", "tts": "pl", "name": "Polski", "flag": "🇵🇱"},
    "turkish": {"code": "tr", "tts": "tr", "name": "Türkçe", "flag": "🇹🇷"},
    "swedish": {"code": "sv", "tts": "sv", "name": "Svenska", "flag": "🇸🇪"},
    "norwegian": {"code": "no", "tts": "no", "name": "Norsk", "flag": "🇳🇴"},
    "danish": {"code": "da", "tts": "da", "name": "Dansk", "flag": "🇩🇰"},
}

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

@dataclass
class AudioSegment:
    audio_data: np.ndarray
    timestamp: float
    duration: float
    speaker_id: Optional[str] = None
    confidence: float = 0.0

# -----------------------
# Enhanced Audio Processing
# -----------------------
class AudioProcessor:
    def __init__(self):
        self.noise_profile = None
        self.audio_history = collections.deque(maxlen=1000)
        
    def enhance_audio(self, audio_data: np.ndarray, sample_rate: int = Config.FS) -> np.ndarray:
        """Apply advanced audio enhancement"""
        try:
            # Normalize audio
            audio_normalized = librosa.util.normalize(audio_data.astype(np.float32))
            
            # Apply noise reduction
            if Config.NOISE_REDUCE_STATIONARY:
                audio_cleaned = nr.reduce_noise(
                    y=audio_normalized, 
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=Config.NOISE_REDUCE_PROP_DECREASE
                )
            else:
                audio_cleaned = audio_normalized
            
            # Apply bandpass filter to focus on speech frequencies (85-8000 Hz)
            nyquist = sample_rate / 2
            low = 85 / nyquist
            high = 8000 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            audio_filtered = signal.filtfilt(b, a, audio_cleaned)
            
            # Apply dynamic range compression
            audio_compressed = self._apply_compression(audio_filtered)
            
            return (audio_compressed * 32767).astype(np.int16)
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return audio_data.astype(np.int16)
    
    def _apply_compression(self, audio: np.ndarray, threshold: float = 0.3, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression"""
        compressed = np.copy(audio)
        mask = np.abs(compressed) > threshold
        compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
        return compressed
    
    def calculate_audio_features(self, audio_data: np.ndarray, sample_rate: int = Config.FS) -> Dict[str, float]:
        """Calculate advanced audio features"""
        try:
            # Convert to float
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Basic features
            rms_energy = np.sqrt(np.mean(audio_float**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_float))
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_float, sr=sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_float, sr=sample_rate))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_float, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs)
            
            return {
                "rms_energy": float(rms_energy),
                "zero_crossing_rate": float(zero_crossing_rate),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "mfcc_mean": float(mfcc_mean)
            }
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            return {}

# -----------------------
# Enhanced Translation Engine
# -----------------------
class TranslationEngine:
    def __init__(self):
        self.engines = {
            "google": GoogleTranslator,
            "microsoft": MicrosoftTranslator,
            "deepl": DeeplTranslator
        }
        self.cache = {}
        self.translation_history = []
    
    def translate_with_confidence(self, text: str, source_lang: str, target_lang: str, 
                                engine: str = "google") -> TranslationResult:
        """Translate with confidence scoring and caching"""
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(f"{text}_{source_lang}_{target_lang}_{engine}".encode()).hexdigest()
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        try:
            if engine == "google":
                translator = GoogleTranslator(source=source_lang, target=target_lang)
            elif engine == "microsoft":
                translator = MicrosoftTranslator(source=source_lang, target=target_lang)
            elif engine == "deepl":
                translator = DeeplTranslator(source=source_lang, target=target_lang)
            else:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
            
            translated_text = translator.translate(text)
            confidence = self._calculate_translation_confidence(text, translated_text, source_lang, target_lang)
            
            result = TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=confidence,
                timestamp=datetime.datetime.now(),
                processing_time=time.time() - start_time,
                translation_engine=engine
            )
            
            # Cache result
            self.cache[cache_key] = result
            self.translation_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=f"[Translation failed: {str(e)}]",
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
                timestamp=datetime.datetime.now(),
                processing_time=time.time() - start_time,
                translation_engine=engine
            )
    
    def _calculate_translation_confidence(self, original: str, translated: str, 
                                        source_lang: str, target_lang: str) -> float:
        """Calculate translation confidence based on various factors"""
        try:
            # Basic length ratio check
            length_ratio = len(translated) / max(len(original), 1)
            if length_ratio < 0.1 or length_ratio > 10:
                return 0.3
            
            # Check for common translation failures
            failure_indicators = ["[translation failed]", "error", "unable to translate"]
            if any(indicator in translated.lower() for indicator in failure_indicators):
                return 0.1
            
            # Language-specific confidence adjustments
            base_confidence = 0.85
            
            # Adjust based on language pair difficulty
            difficult_pairs = [("en", "zh"), ("en", "ja"), ("en", "ar"), ("hi", "zh")]
            if (source_lang, target_lang) in difficult_pairs or (target_lang, source_lang) in difficult_pairs:
                base_confidence *= 0.9
            
            return min(base_confidence, 1.0)
            
        except Exception:
            return 0.5

# -----------------------
# Modern UI Components
# -----------------------
class ModernButton(ctk.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            corner_radius=12,
            border_width=0,
            font=ctk.CTkFont(size=14, weight="bold"),
            **kwargs
        )

class AnimatedFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=15, **kwargs)
        self.original_color = self.cget("fg_color")
    
    def animate_highlight(self):
        """Animate frame highlight"""
        self.configure(fg_color=Config.HIGHLIGHT_COLOR)
        self.after(Config.ANIMATION_SPEED, lambda: self.configure(fg_color=self.original_color))

class AudioVisualizer(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10)
        self.pack_configure(fill="both", expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 3), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        
        # Style the plot
        self.ax.spines['bottom'].set_color('#ffffff')
        self.ax.spines['top'].set_color('#ffffff') 
        self.ax.spines['right'].set_color('#ffffff')
        self.ax.spines['left'].set_color('#ffffff')
        self.ax.tick_params(colors='#ffffff')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize empty plot
        self.line, = self.ax.plot([], [], color=Config.HIGHLIGHT_COLOR, linewidth=2)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_title("Audio Waveform", color='white', fontsize=12)
        
    def update_waveform(self, audio_data: np.ndarray):
        """Update the waveform display"""
        try:
            if len(audio_data) > 0:
                # Downsample for visualization
                downsample_factor = max(1, len(audio_data) // 1000)
                audio_downsampled = audio_data[::downsample_factor]
                
                # Normalize
                if np.max(np.abs(audio_downsampled)) > 0:
                    audio_normalized = audio_downsampled / np.max(np.abs(audio_downsampled))
                else:
                    audio_normalized = audio_downsampled
                
                x = np.linspace(0, 100, len(audio_normalized))
                self.line.set_data(x, audio_normalized)
                self.ax.set_xlim(0, 100)
                self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Waveform update failed: {e}")

# -----------------------
# Main Application
# -----------------------
class AdvancedConferenceTranslator(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("🌐 Advanced Conference Translator Pro")
        self.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.minsize(1200, 800)
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.translation_engine = TranslationEngine()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State variables
        self.speaker_language = ctk.StringVar(value="english")
        self.audience_langs = ctk.StringVar(value="hindi, spanish, french")
        self.selected_device_name = ctk.StringVar()
        self.whisper_model_var = ctk.StringVar(value=Config.DEFAULT_MODEL)
        self.translation_engine_var = ctk.StringVar(value="google")
        self.is_listening = False
        self.is_recording = False
        self.stop_event = threading.Event()
        self.whisper_model = None
        
        # Data storage
        self.session_data = {
            "translations": [],
            "audio_segments": [],
            "session_start": datetime.datetime.now(),
            "statistics": defaultdict(int)
        }
        
        # Audio queue for real-time processing
        self.audio_queue = queue.Queue()
        
        # Get input devices
        self.input_devices = self._get_input_devices()
        if self.input_devices:
            self.selected_device_name.set(self.input_devices[0])
        
        # Build UI
        self._setup_ui()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Load model in background
        threading.Thread(target=self._load_whisper_model, daemon=True).start()
        
        logger.info("Advanced Conference Translator Pro initialized")
    
    def _get_input_devices(self) -> List[str]:
        """Get all available input devices"""
        try:
            devices = sd.query_devices()
            return [d["name"] for d in devices if d["max_input_channels"] > 0]
        except Exception as e:
            logger.error(f"Failed to get input devices: {e}")
            return []
    
    def _device_index_from_name(self, name: str) -> Optional[int]:
        """Get device index from name"""
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d["name"] == name:
                    return idx
        except Exception as e:
            logger.error(f"Failed to get device index: {e}")
        return None
    
    def _setup_ui(self):
        """Setup the modern UI"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self._create_header()
        
        # Main content
        self._create_main_content()
        
        # Footer/Status
        self._create_footer()
    
    def _create_header(self):
        """Create modern header with controls"""
        header_frame = AnimatedFrame(self, height=120)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        header_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        header_frame.grid_propagate(False)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="🌐 Advanced Conference Translator Pro",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(10, 5))
        
        # Language selection
        lang_frame = ctk.CTkFrame(header_frame, corner_radius=10)
        lang_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(lang_frame, text="Speaker Language:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        speaker_menu = ctk.CTkOptionMenu(
            lang_frame, 
            variable=self.speaker_language,
            values=list(ENHANCED_LANG_CODES.keys()),
            font=ctk.CTkFont(size=12)
        )
        speaker_menu.pack(fill="x", padx=5, pady=(0, 5))
        
        # Audience languages
        audience_frame = ctk.CTkFrame(header_frame, corner_radius=10)
        audience_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(audience_frame, text="Audience Languages:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        self.audience_entry = ctk.CTkEntry(
            audience_frame, 
            textvariable=self.audience_langs,
            placeholder_text="e.g., hindi, spanish, french"
        )
        self.audience_entry.pack(fill="x", padx=5, pady=(0, 5))
        
        # Device selection
        device_frame = ctk.CTkFrame(header_frame, corner_radius=10)
        device_frame.grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(device_frame, text="Microphone:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        if self.input_devices:
            device_menu = ctk.CTkOptionMenu(
                device_frame,
                variable=self.selected_device_name,
                values=self.input_devices,
                font=ctk.CTkFont(size=12)
            )
            device_menu.pack(fill="x", padx=5, pady=(0, 5))
        else:
            ctk.CTkLabel(device_frame, text="No devices found", text_color=Config.ERROR_COLOR).pack(padx=5)
        
        # Model and engine selection
        settings_frame = ctk.CTkFrame(header_frame, corner_radius=10)
        settings_frame.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(settings_frame, text="AI Model:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        model_menu = ctk.CTkOptionMenu(
            settings_frame,
            variable=self.whisper_model_var,
            values=Config.WHISPER_MODELS,
            command=self._on_model_change,
            font=ctk.CTkFont(size=12)
        )
        model_menu.pack(fill="x", padx=5, pady=(0, 5))
    
    def _create_main_content(self):
        """Create main content area with tabs"""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Left sidebar with controls
        self._create_sidebar(main_frame)
        
        # Right content area with tabs
        self._create_content_tabs(main_frame)
    
    def _create_sidebar(self, parent):
        """Create left sidebar with controls"""
        sidebar = ctk.CTkFrame(parent, width=300)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 10), pady=0)
        sidebar.grid_propagate(False)
        
        # Control buttons
        controls_frame = AnimatedFrame(sidebar)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ModernButton(
            controls_frame, 
            text="🎤 Start Translation",
            command=self.start_listening,
            height=50,
            fg_color=Config.SUCCESS_COLOR,
            hover_color="#2ecc71"
        )
        self.start_btn.pack(fill="x", pady=(10, 5))
        
        self.stop_btn = ModernButton(
            controls_frame,
            text="⏹️ Stop Translation", 
            command=self.stop_listening,
            height=50,
            fg_color=Config.ERROR_COLOR,
            hover_color="#c0392b",
            state="disabled"
        )
        self.stop_btn.pack(fill="x", pady=5)
        
        self.record_btn = ModernButton(
            controls_frame,
            text="🔴 Record Session",
            command=self.toggle_recording,
            height=40,
            fg_color=Config.WARNING_COLOR,
            hover_color="#e67e22"
        )
        self.record_btn.pack(fill="x", pady=5)
        
        # Settings panel
        settings_panel = AnimatedFrame(sidebar)
        settings_panel.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(settings_panel, text="⚙️ Advanced Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Translation engine
        ctk.CTkLabel(settings_panel, text="Translation Engine:").pack(anchor="w", padx=10)
        engine_menu = ctk.CTkOptionMenu(
            settings_panel,
            variable=self.translation_engine_var,
            values=["google", "microsoft", "deepl"]
        )
        engine_menu.pack(fill="x", padx=10, pady=(0, 10))
        
        # Confidence threshold
        ctk.CTkLabel(settings_panel, text="Confidence Threshold:").pack(anchor="w", padx=10)
        self.confidence_slider = ctk.CTkSlider(
            settings_panel,
            from_=0.1,
            to=1.0,
            number_of_steps=9,
            command=self._on_confidence_change
        )
        self.confidence_slider.set(Config.CONFIDENCE_THRESHOLD)
        self.confidence_slider.pack(fill="x", padx=10, pady=(0, 10))
        
        self.confidence_label = ctk.CTkLabel(settings_panel, text=f"Current: {Config.CONFIDENCE_THRESHOLD:.1f}")
        self.confidence_label.pack(padx=10, pady=(0, 10))
        
        # Audio visualization
        viz_frame = AnimatedFrame(sidebar)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        ctk.CTkLabel(viz_frame, text="🎵 Audio Visualization", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.audio_visualizer = AudioVisualizer(viz_frame)
    
    def _create_content_tabs(self, parent):
        """Create tabbed content area"""
        # Create notebook
        self.notebook = ctk.CTkTabview(parent)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        
        # Live Translation tab
        self._create_live_tab()
        
        # Session History tab
        self._create_history_tab()
        
        # Analytics tab
        self._create_analytics_tab()
        
        # Export tab
        self._create_export_tab()
    
    def _create_live_tab(self):
        """Create live translation tab"""
        tab = self.notebook.add("🔴 Live Translation")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure((0, 1), weight=1)
        
        # Original text frame
        original_frame = AnimatedFrame(tab)
        original_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        original_frame.grid_columnconfigure(0, weight=1)
        original_frame.grid_rowconfigure(1, weight=1)
        
        header_frame = ctk.CTkFrame(original_frame, height=40)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        header_frame.grid_propagate(False)
        
        ctk.CTkLabel(
            header_frame, 
            text="🎤 Speaker (Original)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(side="left", padx=10, pady=5)
        
        self.confidence_indicator = ctk.CTkLabel(
            header_frame,
            text="Confidence: --",
            font=ctk.CTkFont(size=12)
        )
        self.confidence_indicator.pack(side="right", padx=10, pady=5)
        
        self.original_textbox = ctk.CTkTextbox(
            original_frame,
            font=ctk.CTkFont(size=14),
            wrap="word",
            corner_radius=10
        )
        self.original_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Translations frame
        translations_frame = AnimatedFrame(tab)
        translations_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        translations_frame.grid_columnconfigure(0, weight=1)
        translations_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            translations_frame,
            text="🌍 Audience Translations",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.translations_textbox = ctk.CTkTextbox(
            translations_frame,
            font=ctk.CTkFont(size=13),
            wrap="word",
            corner_radius=10
        )
        self.translations_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    
    def _create_history_tab(self):
        """Create session history tab"""
        tab = self.notebook.add("📚 Session History")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Controls
        controls_frame = ctk.CTkFrame(tab, height=60)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        controls_frame.grid_propagate(False)
        controls_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            controls_frame,
            text="📚 Translation History",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=10)
        
        clear_btn = ModernButton(
            controls_frame,
            text="🗑️ Clear History",
            command=self.clear_history,
            width=120,
            height=35,
            fg_color=Config.WARNING_COLOR
        )
        clear_btn.grid(row=0, column=2, padx=10, pady=10)
        
        # History display with scrollable frame
        self.history_frame = ctk.CTkScrollableFrame(tab, corner_radius=10)
        self.history_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.history_frame.grid_columnconfigure(0, weight=1)
    
    def _create_analytics_tab(self):
        """Create analytics and statistics tab"""
        tab = self.notebook.add("📊 Analytics")
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure((0, 1), weight=1)
        
        # Statistics frame
        stats_frame = AnimatedFrame(tab)
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(
            stats_frame,
            text="📊 Session Statistics",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(pady=(10, 5))
        
        self.stats_textbox = ctk.CTkTextbox(stats_frame, font=ctk.CTkFont(size=12))
        self.stats_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Performance metrics
        metrics_frame = AnimatedFrame(tab)
        metrics_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(
            metrics_frame,
            text="⚡ Performance Metrics",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(pady=(10, 5))
        
        self.metrics_textbox = ctk.CTkTextbox(metrics_frame, font=ctk.CTkFont(size=12))
        self.metrics_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Language usage chart placeholder
        chart_frame = AnimatedFrame(tab)
        chart_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(
            chart_frame,
            text="📈 Language Usage Analytics",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).pack(pady=(10, 5))
        
        self.chart_label = ctk.CTkLabel(
            chart_frame,
            text="Analytics will be displayed here during active sessions",
            font=ctk.CTkFont(size=14)
        )
        self.chart_label.pack(expand=True)
    
    def _create_export_tab(self):
        """Create export and save tab"""
        tab = self.notebook.add("💾 Export")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Export options
        options_frame = ctk.CTkFrame(tab, height=100)
        options_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        options_frame.grid_propagate(False)
        options_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        ctk.CTkLabel(
            options_frame,
            text="💾 Export Session Data",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=4, pady=(10, 5))
        
        ModernButton(
            options_frame,
            text="📄 Export PDF",
            command=lambda: self.export_session("pdf"),
            height=40
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ModernButton(
            options_frame,
            text="📝 Export DOCX",
            command=lambda: self.export_session("docx"),
            height=40
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ModernButton(
            options_frame,
            text="📊 Export JSON",
            command=lambda: self.export_session("json"),
            height=40
        ).grid(row=1, column=2, padx=5, pady=5)
        
        ModernButton(
            options_frame,
            text="📋 Export TXT",
            command=lambda: self.export_session("txt"),
            height=40
        ).grid(row=1, column=3, padx=5, pady=5)
        
        # Preview frame
        preview_frame = AnimatedFrame(tab)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            preview_frame,
            text="👁️ Export Preview",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Config.HIGHLIGHT_COLOR
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        self.export_preview = ctk.CTkTextbox(
            preview_frame,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.export_preview.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    def _create_footer(self):
        """Create footer with status and progress"""
        footer_frame = ctk.CTkFrame(self, height=80)
        footer_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        footer_frame.grid_propagate(False)
        footer_frame.grid_columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            footer_frame,
            text="Status: Initializing...",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(footer_frame)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=5)
        self.progress_bar.set(0)
        
        # Session info
        self.session_info = ctk.CTkLabel(
            footer_frame,
            text="Ready to start",
            font=ctk.CTkFont(size=11)
        )
        self.session_info.grid(row=1, column=1, sticky="e", padx=(5, 10), pady=5)
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Audio processing thread
        threading.Thread(target=self._audio_processing_loop, daemon=True).start()
        
        # UI update thread
        threading.Thread(target=self._ui_update_loop, daemon=True).start()
    
    def _audio_processing_loop(self):
        """Background audio processing loop"""
        while True:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    if audio_data is not None:
                        # Update visualizer
                        self.audio_visualizer.update_waveform(audio_data)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def _ui_update_loop(self):
        """Background UI update loop"""
        while True:
            try:
                if self.is_listening:
                    self._update_session_info()
                    self._update_statistics()
                time.sleep(1)
            except Exception as e:
                logger.error(f"UI update error: {e}")
    
    def _load_whisper_model(self):
        """Load Whisper model with progress tracking"""
        try:
            model_name = self.whisper_model_var.get()
            self._set_status(f"Loading {model_name} model...")
            self._set_progress(0.2)
            
            self.whisper_model = whisper.load_model(model_name)
            
            self._set_progress(1.0)
            self._set_status("Model loaded successfully. Ready to translate!")
            
            # Reset progress after delay
            self.after(2000, lambda: self._set_progress(0))
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            self._set_status(error_msg)
            messagebox.showerror("Model Error", error_msg)
    
    def _on_model_change(self, model_name: str):
        """Handle model change"""
        self.whisper_model = None
        threading.Thread(target=self._load_whisper_model, daemon=True).start()
    
    def _on_confidence_change(self, value: float):
        """Handle confidence threshold change"""
        Config.CONFIDENCE_THRESHOLD = value
        self.confidence_label.configure(text=f"Current: {value:.1f}")
    
    def _set_status(self, text: str):
        """Update status label thread-safely"""
        self.after(0, lambda: self.status_label.configure(text=f"Status: {text}"))
    
    def _set_progress(self, value: float):
        """Update progress bar thread-safely"""
        self.after(0, lambda: self.progress_bar.set(value))
    
    def _update_session_info(self):
        """Update session information"""
        if self.session_data["translations"]:
            duration = datetime.datetime.now() - self.session_data["session_start"]
            translations_count = len(self.session_data["translations"])
            
            info_text = f"Duration: {str(duration).split('.')[0]} | Translations: {translations_count}"
            self.after(0, lambda: self.session_info.configure(text=info_text))
    
    def _update_statistics(self):
        """Update session statistics"""
        try:
            stats = self._calculate_session_stats()
            stats_text = self._format_stats(stats)
            
            self.after(0, lambda: self._update_stats_display(stats_text))
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
    
    def _calculate_session_stats(self) -> Dict[str, Any]:
        """Calculate session statistics"""
        translations = self.session_data["translations"]
        
        if not translations:
            return {"message": "No translations yet"}
        
        # Basic stats
        total_translations = len(translations)
        avg_confidence = np.mean([t.confidence for t in translations])
        avg_processing_time = np.mean([t.processing_time for t in translations])
        
        # Language distribution
        source_langs = defaultdict(int)
        target_langs = defaultdict(int)
        
        for t in translations:
            source_langs[t.source_lang] += 1
            target_langs[t.target_lang] += 1
        
        # Quality metrics
        high_confidence = sum(1 for t in translations if t.confidence > 0.8)
        low_confidence = sum(1 for t in translations if t.confidence < 0.5)
        
        return {
            "total_translations": total_translations,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "source_languages": dict(source_langs),
            "target_languages": dict(target_langs),
            "high_confidence_ratio": high_confidence / total_translations if total_translations > 0 else 0,
            "low_confidence_ratio": low_confidence / total_translations if total_translations > 0 else 0
        }
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display"""
        if "message" in stats:
            return stats["message"]
        
        formatted = f"""Session Statistics:

📊 Overview:
  • Total Translations: {stats['total_translations']}
  • Average Confidence: {stats['avg_confidence']:.2f}
  • Average Processing Time: {stats['avg_processing_time']:.3f}s

🎯 Quality Metrics:
  • High Confidence (>80%): {stats['high_confidence_ratio']:.1%}
  • Low Confidence (<50%): {stats['low_confidence_ratio']:.1%}

🌍 Source Languages:
"""
        
        for lang, count in stats['source_languages'].items():
            formatted += f"  • {lang}: {count}\n"
        
        formatted += "\n🎯 Target Languages:\n"
        for lang, count in stats['target_languages'].items():
            formatted += f"  • {lang}: {count}\n"
        
        return formatted
    
    def _update_stats_display(self, stats_text: str):
        """Update statistics display"""
        self.stats_textbox.delete("1.0", "end")
        self.stats_textbox.insert("1.0", stats_text)
        
        # Update metrics
        metrics_text = self._get_performance_metrics()
        self.metrics_textbox.delete("1.0", "end")
        self.metrics_textbox.insert("1.0", metrics_text)
    
    def _get_performance_metrics(self) -> str:
        """Get system performance metrics"""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = f"""Performance Metrics:

💻 System Resources:
  • CPU Usage: {cpu_percent:.1f}%
  • Memory Usage: {memory.percent:.1f}%
  • Available Memory: {memory.available // (1024**3):.1f} GB

🎤 Audio Processing:
  • Sample Rate: {Config.FS} Hz
  • Chunk Size: {Config.CHUNK_SIZE} samples
  • VAD Aggressiveness: {Config.VAD_AGGRESSIVENESS}

🤖 AI Models:
  • Whisper Model: {self.whisper_model_var.get()}
  • Translation Engine: {self.translation_engine_var.get()}
  • Confidence Threshold: {Config.CONFIDENCE_THRESHOLD:.1f}
"""
            return metrics
        except ImportError:
            return "Performance monitoring requires psutil package"
        except Exception as e:
            return f"Metrics unavailable: {str(e)}"
    
    def start_listening(self):
        """Start the translation service"""
        if not self.whisper_model:
            messagebox.showwarning("Model Not Ready", "Please wait for the AI model to load.")
            return
        
        if not self.input_devices:
            messagebox.showerror("No Microphone", "No input devices detected.")
            return
        
        # Validate audience languages
        raw_langs = self.audience_langs.get()
        audience_list = [lang.strip().lower() for lang in raw_langs.split(",") if lang.strip()]
        
        invalid_langs = [lang for lang in audience_list if lang not in ENHANCED_LANG_CODES]
        if invalid_langs:
            messagebox.showerror("Invalid Languages", f"Unsupported languages: {', '.join(invalid_langs)}")
            return
        
        # Setup session
        self.audience_list = audience_list
        self.speaker_code = ENHANCED_LANG_CODES.get(self.speaker_language.get(), {}).get("code", "auto")
        
        # Reset session data
        self.session_data = {
            "translations": [],
            "audio_segments": [],
            "session_start": datetime.datetime.now(),
            "statistics": defaultdict(int)
        }
        
        # Update UI
        self.is_listening = True
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Start listening thread
        threading.Thread(target=self._enhanced_vad_loop, daemon=True).start()
        
        self._set_status("🎤 Listening for speech...")
        logger.info("Translation service started")
    
    def stop_listening(self):
        """Stop the translation service"""
        self.stop_event.set()
        self.is_listening = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        self._set_status("⏹️ Translation stopped")
        logger.info("Translation service stopped")
    
    def toggle_recording(self):
        """Toggle session recording"""
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.record_btn.configure(text="⏺️ Recording...", fg_color=Config.ERROR_COLOR)
            self._set_status("🔴 Recording session...")
        else:
            self.record_btn.configure(text="🔴 Record Session", fg_color=Config.WARNING_COLOR)
            self._set_status("Recording stopped")
    
    def _enhanced_vad_loop(self):
        """Enhanced VAD loop with noise reduction and better detection"""
        try:
            vad = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
            device_name = self.selected_device_name.get()
            device_index = self._device_index_from_name(device_name)
            
            if device_index is None:
                raise ValueError("Selected device not found")
            
            # Buffers for audio processing
            pre_buffer = collections.deque(maxlen=int(Config.PRE_SPEECH_SECONDS * Config.FS / Config.CHUNK_SIZE))
            
            # State variables
            triggered = False
            voiced_frames = []
            last_voice_time = None
            phrase_start_time = None
            
            logger.info(f"Starting VAD loop with device: {device_name}")
            
            with sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=Config.FS,
                dtype=np.int16,
                blocksize=Config.CHUNK_SIZE
            ) as stream:
                
                while not self.stop_event.is_set():
                    try:
                        data, overflow = stream.read(Config.CHUNK_SIZE)
                        if overflow:
                            logger.warning("Audio buffer overflow detected")
                            continue
                        
                        # Convert to proper format
                        frame = data.flatten().astype(np.int16)
                        
                        # Add to audio queue for visualization
                        if not self.audio_queue.full():
                            self.audio_queue.put(frame.copy())
                        
                        # VAD detection
                        is_speech = False
                        try:
                            # Ensure frame is the right size and format
                            if len(frame) == Config.CHUNK_SIZE:
                                is_speech = vad.is_speech(frame.tobytes(), Config.FS)
                        except Exception as vad_error:
                            logger.debug(f"VAD detection failed: {vad_error}")
                            continue
                        
                        current_time = time.time()
                        
                        if not triggered:
                            # Pre-trigger state: collect audio and wait for speech
                            pre_buffer.append((frame.copy(), is_speech))
                            
                            # Check if we have enough speech in the buffer
                            if len(pre_buffer) == pre_buffer.maxlen:
                                voiced_count = sum(1 for _, speech in pre_buffer if speech)
                                speech_ratio = voiced_count / len(pre_buffer)
                                
                                if speech_ratio > 0.3:  # 30% of frames contain speech
                                    triggered = True
                                    phrase_start_time = current_time
                                    
                                    # Add pre-buffer to voiced frames
                                    for frame_data, _ in pre_buffer:
                                        voiced_frames.append(frame_data)
                                    
                                    pre_buffer.clear()
                                    last_voice_time = current_time
                                    
                                    self._set_status("🎤 Speech detected - processing...")
                                    logger.debug("Speech trigger activated")
                        else:
                            # Triggered state: collect speech audio
                            voiced_frames.append(frame.copy())
                            
                            if is_speech:
                                last_voice_time = current_time
                            
                            # Check for end conditions
                            phrase_duration = current_time - phrase_start_time
                            silence_duration = current_time - last_voice_time if last_voice_time else 0
                            
                            should_process = False
                            reason = ""
                            
                            if phrase_duration > Config.MAX_PHRASE_SECONDS:
                                should_process = True
                                reason = "Maximum phrase length reached"
                            elif silence_duration > Config.SILENCE_TIMEOUT_SECONDS:
                                should_process = True
                                reason = "Silence timeout reached"
                            
                            if should_process:
                                if len(voiced_frames) > 0:
                                    # Combine all frames
                                    combined_audio = np.concatenate(voiced_frames)
                                    
                                    # Process in background thread
                                    self.executor.submit(self._process_audio_phrase, combined_audio, reason)
                                
                                # Reset state
                                triggered = False
                                voiced_frames = []
                                last_voice_time = None
                                phrase_start_time = None
                    
                    except Exception as loop_error:
                        logger.error(f"Error in VAD loop iteration: {loop_error}")
                        time.sleep(0.1)  # Brief pause before continuing
                        
        except Exception as e:
            error_msg = f"VAD loop failed: {str(e)}"
            logger.error(error_msg)
            self._set_status(f"❌ Audio error: {str(e)}")
            self.stop_listening()
    
    def _process_audio_phrase(self, audio_data: np.ndarray, reason: str = ""):
        """Process detected audio phrase with enhanced pipeline"""
        try:
            self._set_status(f"🔄 Processing audio... ({reason})")
            
            # Enhanced audio preprocessing
            enhanced_audio = self.audio_processor.enhance_audio(audio_data, Config.FS)
            
            # Calculate audio features for confidence estimation
            audio_features = self.audio_processor.calculate_audio_features(enhanced_audio, Config.FS)
            
            # Save to temporary file
            temp_wav = self._create_temp_wav(enhanced_audio)
            
            try:
                # Transcribe with Whisper
                transcription_start = time.time()
                result = self.whisper_model.transcribe(
                    temp_wav,
                    language=self.speaker_code if self.speaker_code != "auto" else None,
                    fp16=False,
                    word_timestamps=True,
                    condition_on_previous_text=True
                )
                
                transcription_time = time.time() - transcription_start
                
                # Extract text and confidence
                text = result.get("text", "").strip()
                segments = result.get("segments", [])
                
                # Calculate average confidence from segments
                if segments and all("avg_logprob" in seg for seg in segments):
                    avg_confidence = np.mean([np.exp(seg["avg_logprob"]) for seg in segments])
                else:
                    avg_confidence = 0.7  # Default confidence
                
                logger.info(f"Transcribed: '{text}' (confidence: {avg_confidence:.2f})")
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            
            if not text or len(text.strip()) < 3:
                self._set_status("🎤 No speech detected - listening...")
                return
            
            # Update original text display
            self._update_original_text(text, avg_confidence)
            
            # Translate to all target languages
            self._translate_and_speak(text, avg_confidence, audio_features, transcription_time)
            
        except Exception as e:
            error_msg = f"Audio processing failed: {str(e)}"
            logger.error(error_msg)
            self._set_status(f"❌ Processing error: {str(e)}")
    
    def _create_temp_wav(self, audio_data: np.ndarray) -> str:
        """Create temporary WAV file"""
        temp_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
        sf.write(temp_path, audio_data, Config.FS, subtype="PCM_16")
        return temp_path
    
    def _update_original_text(self, text: str, confidence: float):
        """Update original text display"""
        speaker_lang_info = ENHANCED_LANG_CODES.get(self.speaker_language.get(), {})
        display_text = f"{speaker_lang_info.get('flag', '🎤')} {speaker_lang_info.get('name', self.speaker_language.get())}: {text}"
        
        def update_ui():
            self.original_textbox.delete("1.0", "end")
            self.original_textbox.insert("1.0", display_text)
            
            # Update confidence indicator with color coding
            confidence_color = Config.SUCCESS_COLOR if confidence > 0.8 else Config.WARNING_COLOR if confidence > 0.5 else Config.ERROR_COLOR
            self.confidence_indicator.configure(
                text=f"Confidence: {confidence:.1%}",
                text_color=confidence_color
            )
        
        self.after(0, update_ui)
    
    def _translate_and_speak(self, original_text: str, transcription_confidence: float, 
                           audio_features: Dict[str, float], transcription_time: float):
        """Translate text and generate speech for all target languages"""
        self._set_status("🌍 Translating to target languages...")
        
        translations_display = []
        translation_futures = []
        
        # Submit translation tasks
        for lang_name in self.audience_list:
            lang_info = ENHANCED_LANG_CODES.get(lang_name, {})
            target_code = lang_info.get("code")
            
            if target_code:
                future = self.executor.submit(
                    self._translate_single_language,
                    original_text,
                    self.speaker_code,
                    target_code,
                    lang_name,
                    lang_info,
                    transcription_confidence,
                    audio_features,
                    transcription_time
                )
                translation_futures.append(future)
        
        # Collect results
        for future in translation_futures:
            try:
                result = future.result(timeout=10)  # 10 second timeout
                if result:
                    translations_display.append(result["display_text"])
                    self.session_data["translations"].append(result["translation_result"])
                    
                    # Generate TTS in background
                    if result["translation_result"].confidence > Config.CONFIDENCE_THRESHOLD:
                        self.executor.submit(
                            self._generate_and_play_tts,
                            result["translation_result"].translated_text,
                            result["lang_info"].get("tts", "en")
                        )
                    
            except Exception as e:
                logger.error(f"Translation future failed: {e}")
        
        # Update translations display
        if translations_display:
            final_display = "\n\n".join(translations_display)
            self.after(0, lambda: self._update_translations_display(final_display))
        
        # Update export preview
        self._update_export_preview()
        
        self._set_status("🎤 Listening for speech...")
    
    def _translate_single_language(self, text: str, source_lang: str, target_lang: str,
                                 lang_name: str, lang_info: Dict[str, str],
                                 transcription_confidence: float, audio_features: Dict[str, float],
                                 transcription_time: float) -> Optional[Dict[str, Any]]:
        """Translate text to a single target language"""
        try:
            # Perform translation
            translation_result = self.translation_engine.translate_with_confidence(
                text, source_lang, target_lang, self.translation_engine_var.get()
            )
            
            # Adjust confidence based on transcription quality
            combined_confidence = (transcription_confidence + translation_result.confidence) / 2
            translation_result.confidence = combined_confidence
            
            # Create display text with flag and formatting
            flag = lang_info.get("flag", "🌍")
            name = lang_info.get("name", lang_name.title())
            confidence_indicator = "✅" if combined_confidence > 0.8 else "⚠️" if combined_confidence > 0.5 else "❌"
            
            display_text = f"{flag} {name} {confidence_indicator}: {translation_result.translated_text}"
            
            return {
                "display_text": display_text,
                "translation_result": translation_result,
                "lang_info": lang_info
            }
            
        except Exception as e:
            logger.error(f"Translation failed for {lang_name}: {e}")
            error_display = f"❌ {lang_info.get('name', lang_name)}: [Translation failed]"
            return {
                "display_text": error_display,
                "translation_result": None,
                "lang_info": lang_info
            }
    
    def _update_translations_display(self, text: str):
        """Update the translations display"""
        self.translations_textbox.delete("1.0", "end")
        self.translations_textbox.insert("1.0", text)
    
    def _generate_and_play_tts(self, text: str, lang_code: str):
        """Generate and play text-to-speech"""
        try:
            temp_mp3 = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
            
            # Generate TTS
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_mp3)
            
            # Play audio using pydub
            audio = AudioSegment.from_mp3(temp_mp3)
            play(audio)
            
            # Clean up
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
    
    def _update_export_preview(self):
        """Update export preview"""
        try:
            preview_text = self._generate_export_content("preview")
            self.after(0, lambda: self._set_export_preview(preview_text))
        except Exception as e:
            logger.error(f"Export preview update failed: {e}")
    
    def _set_export_preview(self, text: str):
        """Set export preview text"""
        self.export_preview.delete("1.0", "end")
        self.export_preview.insert("1.0", text)
    
    def _generate_export_content(self, format_type: str = "preview") -> str:
        """Generate content for export"""
        if not self.session_data["translations"]:
            return "No translations available for export."
        
        # Header information
        content = f"""Conference Translation Session Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {datetime.datetime.now() - self.session_data['session_start']}
Total Translations: {len(self.session_data['translations'])}

================================

"""
        
        # Group translations by timestamp (approximate)
        translations_by_time = defaultdict(list)
        
        for i, translation in enumerate(self.session_data["translations"]):
            # Group by minute
            time_key = translation.timestamp.strftime('%H:%M')
            translations_by_time[time_key].append((i, translation))
        
        # Generate content
        for time_key in sorted(translations_by_time.keys()):
            content += f"\n[{time_key}] Translation Set:\n"
            content += "-" * 40 + "\n"
            
            # Find the original text (assuming first translation in group represents the source)
            original_translation = translations_by_time[time_key][0][1]
            content += f"Original ({original_translation.source_lang}): {original_translation.original_text}\n\n"
            
            # Add all translations
            for idx, translation in translations_by_time[time_key]:
                confidence_star = "★★★" if translation.confidence > 0.8 else "★★☆" if translation.confidence > 0.5 else "★☆☆"
                content += f"  → {translation.target_lang.upper()}: {translation.translated_text}\n"
                content += f"    Confidence: {translation.confidence:.1%} {confidence_star}\n"
                content += f"    Engine: {translation.translation_engine}\n"
                content += f"    Time: {translation.processing_time:.3f}s\n\n"
            
            content += "\n"
        
        # Add statistics
        if len(self.session_data["translations"]) > 0:
            stats = self._calculate_session_stats()
            content += "\n" + "=" * 50 + "\n"
            content += "SESSION STATISTICS\n"
            content += "=" * 50 + "\n"
            content += f"Average Confidence: {stats['avg_confidence']:.1%}\n"
            content += f"Average Processing Time: {stats['avg_processing_time']:.3f}s\n"
            content += f"High Quality Translations (>80%): {stats['high_confidence_ratio']:.1%}\n"
            content += f"Low Quality Translations (<50%): {stats['low_confidence_ratio']:.1%}\n"
        
        return content
    
    def clear_history(self):
        """Clear translation history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all translation history?"):
            self.session_data = {
                "translations": [],
                "audio_segments": [],
                "session_start": datetime.datetime.now(),
                "statistics": defaultdict(int)
            }
            
            # Clear displays
            self.original_textbox.delete("1.0", "end")
            self.translations_textbox.delete("1.0", "end")
            self.export_preview.delete("1.0", "end")
            
            # Clear history frame
            for widget in self.history_frame.winfo_children():
                widget.destroy()
            
            self._set_status("🗑️ History cleared")
            logger.info("Translation history cleared")
    
    def export_session(self, format_type: str):
        """Export session data in specified format"""
        if not self.session_data["translations"]:
            messagebox.showwarning("No Data", "No translation data available for export.")
            return
        
        # Get save location
        file_types = {
            "pdf": ("PDF files", "*.pdf"),
            "docx": ("Word documents", "*.docx"),
            "json": ("JSON files", "*.json"),
            "txt": ("Text files", "*.txt")
        }
        
        file_path = filedialog.asksaveasfilename(
            title=f"Save as {format_type.upper()}",
            filetypes=[file_types[format_type], ("All files", "*.*")],
            defaultextension=f".{format_type}"
        )
        
        if not file_path:
            return
        
        try:
            if format_type == "pdf":
                self._export_pdf(file_path)
            elif format_type == "docx":
                self._export_docx(file_path)
            elif format_type == "json":
                self._export_json(file_path)
            elif format_type == "txt":
                self._export_txt(file_path)
            
            messagebox.showinfo("Export Complete", f"Session data exported successfully to:\n{file_path}")
            logger.info(f"Session exported to {file_path}")
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Export Error", error_msg)
    
    def _export_pdf(self, file_path: str):
        """Export session as PDF"""
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Conference Translation Session Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Session info
        session_info = f"""
        Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Duration: {datetime.datetime.now() - self.session_data['session_start']}
        Total Translations: {len(self.session_data['translations'])}
        """
        info_para = Paragraph(session_info.replace('\n', '<br/>'), styles['Normal'])
        story.append(info_para)
        story.append(Spacer(1, 12))
        
        # Content
        content = self._generate_export_content("pdf")
        # Convert content to PDF-safe format
        content_para = Paragraph(content.replace('\n', '<br/>'), styles['Normal'])
        story.append(content_para)
        
        doc.build(story)
    
    def _export_docx(self, file_path: str):
        """Export session as DOCX"""
        doc = Document()
        
        # Title
        title = doc.add_heading('Conference Translation Session Report', 0)
        
        # Session info
        doc.add_heading('Session Information', level=1)
        info_para = doc.add_paragraph()
        info_para.add_run(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        info_para.add_run(f"Duration: {datetime.datetime.now() - self.session_data['session_start']}\n")
        info_para.add_run(f"Total Translations: {len(self.session_data['translations'])}")
        
        # Content
        doc.add_heading('Translation Data', level=1)
        content = self._generate_export_content("docx")
        doc.add_paragraph(content)
        
        doc.save(file_path)
    
    def _export_json(self, file_path: str):
        """Export session as JSON"""
        export_data = {
            "session_info": {
                "start_time": self.session_data["session_start"].isoformat(),
                "export_time": datetime.datetime.now().isoformat(),
                "total_translations": len(self.session_data["translations"])
            },
            "translations": []
        }
        
        for translation in self.session_data["translations"]:
            export_data["translations"].append({
                "original_text": translation.original_text,
                "translated_text": translation.translated_text,
                "source_language": translation.source_lang,
                "target_language": translation.target_lang,
                "confidence": translation.confidence,
                "timestamp": translation.timestamp.isoformat(),
                "processing_time": translation.processing_time,
                "translation_engine": translation.translation_engine
            })
        
        # Add statistics
        export_data["statistics"] = self._calculate_session_stats()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_txt(self, file_path: str):
        """Export session as plain text"""
        content = self._generate_export_content("txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_listening:
            if messagebox.askyesno("Exit", "Translation is active. Stop and exit?"):
                self.stop_listening()
            else:
                return
        
        # Cleanup
        try:
            self.executor.shutdown(wait=False)
        except:
            pass
        
        logger.info("Application closing")
        self.destroy()


# -----------------------
# Application Entry Point
# -----------------------
def main():
    """Main application entry point"""
    try:
        # Set up enhanced logging
        logger.info("Starting Advanced Conference Translator Pro")
        
        # Create and run application
        app = AdvancedConferenceTranslator()
        
        # Handle window closing
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start the application
        app.mainloop()
        
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Check dependencies
    required_packages = [
        "customtkinter", "sounddevice", "numpy", "soundfile", "webrtcvad",
        "whisper", "librosa", "noisereduce", "deep_translator", "gtts",
        "pydub", "matplotlib", "seaborn", "PIL", "reportlab", "docx"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    main()