import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import joblib
import pyttsx3
import os
import json
from collections import deque, Counter
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- ENHANCED CONFIG ----------------
MODEL_PATH = 'sign_language_model.pkl'
DATA_DIR = "static_gesture_data"
CONFIG_FILE = 'translator_config.json'
HISTORY_FILE = 'translation_history.json'

# Default configuration
DEFAULT_CONFIG = {
    'confidence_threshold': 0.3,
    'cam_width': 640,
    'cam_height': 480,
    'prediction_window_size': 20,
    'stability_threshold': 0.7,
    'tts_display_interval': 0.8,
    'auto_save_history': True,
    'show_confidence_bars': True,
    'enable_gesture_smoothing': True,
    'dark_theme': True,
    'tts_rate': 150
}

# Load or create configuration
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = {**DEFAULT_CONFIG, **json.load(f)}
else:
    config = DEFAULT_CONFIG.copy()

# UI Constants
CAM_WIDTH, CAM_HEIGHT = config['cam_width'], config['cam_height']
LANDMARK_CANVAS_SIZE = 480
UI_HEIGHT = 280
WINDOW_WIDTH = CAM_WIDTH + LANDMARK_CANVAS_SIZE
WINDOW_HEIGHT = CAM_HEIGHT + UI_HEIGHT
TTS_DISPLAY_AREA_HEIGHT = 180
TTS_IMAGE_SIZE = 120

# Theme colors
if config['dark_theme']:
    UI_BG_COLOR = (45, 45, 45)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (70, 70, 70)
    ACTIVE_COLOR = (100, 200, 100)
else:
    UI_BG_COLOR = (240, 240, 240)
    TEXT_COLOR = (0, 0, 0)
    BUTTON_COLOR = (200, 200, 200)
    ACTIVE_COLOR = (100, 150, 255)

# ---------------- ENHANCED GLOBAL STATE ----------------
class TranslatorState:
    def __init__(self):
        # Gesture Recognition
        self.sentence = []
        self.last_stable_char = None
        self.prediction_history = deque(maxlen=config['prediction_window_size'])
        self.confidence_history = deque(maxlen=config['prediction_window_size'])
        
        # Statistics
        self.session_start_time = time.time()
        self.total_gestures_recognized = 0
        self.accuracy_scores = []
        
        # Text-to-Sign
        self.input_text = ""
        self.is_text_input_active = False
        self.text_to_sign_sentence = ""
        self.tts_char_index = 0
        self.tts_last_update_time = 0
        self.tts_paused = False
        
        # UI State
        self.show_settings = False
        self.show_history = False
        self.selected_tab = 'translate'  # 'translate', 'history', 'settings'
        
        # Translation History
        self.translation_history = self.load_history()
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_history(self):
        if config['auto_save_history']:
            try:
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(self.translation_history[-100:], f, indent=2)  # Keep last 100
            except Exception as e:
                logger.error(f"Failed to save history: {e}")

state = TranslatorState()
sign_images = {}

# ---------------- ENHANCED MEDIAPIPE & MODEL ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Enhanced model loading with error handling
try:
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ---------------- ENHANCED FUNCTIONS ----------------
def save_config():
    """Save current configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def load_sign_images():
    """Enhanced image loading with better error handling and preprocessing"""
    global sign_images
    sign_images.clear()
    
    if not os.path.exists(DATA_DIR):
        logger.warning(f"Data directory {DATA_DIR} not found")
        return
    
    loaded_count = 0
    for label in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(class_dir):
            # Find the best quality image
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if image_files:
                # Sort by file size to get potentially better quality images
                image_files.sort(key=lambda x: os.path.getsize(os.path.join(class_dir, x)), reverse=True)
                
                for img_file in image_files[:3]:  # Try up to 3 images
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Enhanced preprocessing
                        img = cv2.resize(img, (TTS_IMAGE_SIZE, TTS_IMAGE_SIZE))
                        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)  # Enhance contrast
                        sign_images[label.upper()] = img
                        loaded_count += 1
                        break
    
    logger.info(f"Loaded {loaded_count} sample sign images")

def enhanced_speak(text):
    """Enhanced TTS with better error handling and customization"""
    if not text.strip():
        return
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', config['tts_rate'])
        
        # Try to set a more natural voice
        voices = engine.getProperty('voices')
        if voices:
            for voice in voices:
                if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        logger.error(f"TTS failed: {e}")

def recognize_gesture_enhanced(landmarks_list):
    """Enhanced gesture recognition with smoothing and confidence analysis"""
    if not landmarks_list or model is None:
        return None, 0.0, {}
    
    try:
        landmarks_row = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list]).flatten()
        
        # Get prediction probabilities
        probs = model.predict_proba([landmarks_row])[0]
        prediction = model.predict([landmarks_row])[0]
        confidence = np.max(probs)
        
        # Get top 3 predictions for analysis
        top_indices = np.argsort(probs)[-3:][::-1]
        top_predictions = {
            model.classes_[i]: probs[i] for i in top_indices
        }
        
        return prediction, confidence, top_predictions
    except Exception as e:
        logger.error(f"Gesture recognition failed: {e}")
        return None, 0.0, {}

def draw_enhanced_landmarks(landmarks, image_size=LANDMARK_CANVAS_SIZE):
    """Enhanced landmark visualization with better colors and connections"""
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if config['dark_theme']:
        canvas.fill(30)
    else:
        canvas.fill(255)
    
    if not landmarks:
        return canvas

    # Normalize and scale landmarks
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Add padding
    padding = 0.1
    range_coords = max_coords - min_coords
    min_coords -= range_coords * padding
    max_coords += range_coords * padding
    range_coords = max_coords - min_coords
    
    scale = min(image_size / range_coords[0], image_size / range_coords[1]) * 0.8
    offset = (image_size - range_coords * scale) / 2 - min_coords * scale

    # Draw connections with different colors for different parts
    connection_colors = {
        'thumb': (255, 100, 100),
        'index': (100, 255, 100),
        'middle': (100, 100, 255),
        'ring': (255, 255, 100),
        'pinky': (255, 100, 255),
        'palm': (200, 200, 200)
    }
    
    # Define finger groups
    finger_connections = {
        'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
        'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
        'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
        'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
        'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
    }
    
    # Draw connections
    for finger, connections in finger_connections.items():
        color = connection_colors.get(finger, (255, 255, 255))
        for idx1, idx2 in connections:
            pt1 = (int(landmarks[idx1].x * scale + offset[0]), 
                   int(landmarks[idx1].y * scale + offset[1]))
            pt2 = (int(landmarks[idx2].x * scale + offset[0]), 
                   int(landmarks[idx2].y * scale + offset[1]))
            cv2.line(canvas, pt1, pt2, color, 2)

    # Draw landmarks
    for i, landmark in enumerate(landmarks):
        center = (int(landmark.x * scale + offset[0]), 
                 int(landmark.y * scale + offset[1]))
        
        # Different colors for different landmark types
        if i == 0:  # Wrist
            color = (255, 255, 255)
            radius = 6
        elif i in [4, 8, 12, 16, 20]:  # Fingertips
            color = (0, 255, 255)
            radius = 5
        else:  # Other landmarks
            color = (100, 200, 255)
            radius = 3
            
        cv2.circle(canvas, center, radius, color, -1)
        cv2.circle(canvas, center, radius + 1, (0, 0, 0), 1)
    
    return canvas

def draw_confidence_meter(frame, confidence, x, y, width=200, height=20):
    """Draw a confidence meter bar"""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Confidence bar
    conf_width = int(width * confidence)
    if confidence > 0.7:
        color = (0, 255, 0)
    elif confidence > 0.4:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    
    cv2.rectangle(frame, (x, y), (x + conf_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), TEXT_COLOR, 2)
    
    # Text
    cv2.putText(frame, f"{confidence:.1%}", (x + width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

def draw_top_predictions(frame, top_predictions, x, y):
    """Draw top 3 predictions with confidence"""
    if not top_predictions:
        return
    
    cv2.putText(frame, "Top Predictions:", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    
    for i, (pred, conf) in enumerate(top_predictions.items()):
        y_pos = y + 25 + i * 20
        cv2.putText(frame, f"{pred}: {conf:.2%}", (x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

# ---------------- ENHANCED MOUSE CALLBACKS ----------------
def get_button_rects():
    """Get button rectangles based on current window size"""
    return {
        'clear': (WINDOW_WIDTH - 200, CAM_HEIGHT + 40, 150, 35),
        'speak': (WINDOW_WIDTH - 200, CAM_HEIGHT + 85, 150, 35),
        'pause_tts': (WINDOW_WIDTH - 200, CAM_HEIGHT + 130, 150, 35),
        'settings': (WINDOW_WIDTH - 200, CAM_HEIGHT + 175, 150, 35),
        'text_input': (30, CAM_HEIGHT + UI_HEIGHT + 20, WINDOW_WIDTH - 260, 40),
        'tab_translate': (30, CAM_HEIGHT + 15, 100, 25),
        'tab_history': (140, CAM_HEIGHT + 15, 100, 25),
    }

def enhanced_mouse_callback(event, x, y, flags, param):
    """Enhanced mouse callback with multiple UI elements"""
    global state, config
    
    if event == cv2.EVENT_LBUTTONDOWN:
        rects = get_button_rects()
        
        # Tab switching
        if point_in_rect((x, y), rects['tab_translate']):
            state.selected_tab = 'translate'
        elif point_in_rect((x, y), rects['tab_history']):
            state.selected_tab = 'history'
        
        # Main buttons
        elif point_in_rect((x, y), rects['clear']):
            state.sentence = []
            state.last_stable_char = None
            state.prediction_history.clear()
            logger.info("Sentence cleared")
            
        elif point_in_rect((x, y), rects['speak']):
            full_sentence = "".join(state.sentence)
            if full_sentence.strip():
                # Add to history
                state.translation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'sign_to_text',
                    'content': full_sentence,
                    'confidence': np.mean(list(state.confidence_history)) if state.confidence_history else 0
                })
                state.save_history()
                threading.Thread(target=enhanced_speak, args=(full_sentence,), daemon=True).start()
        
        elif point_in_rect((x, y), rects['pause_tts']):
            state.tts_paused = not state.tts_paused
            
        elif point_in_rect((x, y), rects['settings']):
            state.show_settings = not state.show_settings
            
        elif point_in_rect((x, y), rects['text_input']):
            state.is_text_input_active = True
            state.input_text = ""
            state.text_to_sign_sentence = ""
        else:
            state.is_text_input_active = False

def point_in_rect(point, rect):
    """Check if point is inside rectangle"""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

# ---------------- ENHANCED UI DRAWING ----------------
def draw_enhanced_ui(ui_frame, current_char, current_conf, top_predictions):
    """Draw the enhanced user interface"""
    rects = get_button_rects()
    
    # Draw tabs
    tabs = [('translate', 'TRANSLATE'), ('history', 'HISTORY')]
    for i, (tab_id, tab_name) in enumerate(tabs):
        rect = rects[f'tab_{tab_id}']
        color = ACTIVE_COLOR if state.selected_tab == tab_id else BUTTON_COLOR
        cv2.rectangle(ui_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, -1)
        cv2.putText(ui_frame, tab_name, (rect[0]+10, rect[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    if state.selected_tab == 'translate':
        draw_translate_tab(ui_frame, current_char, current_conf, top_predictions, rects)
    elif state.selected_tab == 'history':
        draw_history_tab(ui_frame, rects)

def draw_translate_tab(ui_frame, current_char, current_conf, top_predictions, rects):
    """Draw the main translation interface"""
    # Current detection
    cv2.putText(ui_frame, f"Detection: {current_char}", (30, CAM_HEIGHT + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    # Confidence meter
    if config['show_confidence_bars'] and current_conf > 0:
        draw_confidence_meter(ui_frame, current_conf, 30, CAM_HEIGHT + 85)
    
    # Current sentence
    sentence_text = ''.join(state.sentence).upper()
    cv2.putText(ui_frame, f"Sentence: {sentence_text}", (30, CAM_HEIGHT + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    # Top predictions
    if top_predictions and len(top_predictions) > 1:
        draw_top_predictions(ui_frame, top_predictions, 30, CAM_HEIGHT + 140)
    
    # Statistics
    session_time = int(time.time() - state.session_start_time)
    stats_text = f"Session: {session_time//60}m {session_time%60}s | Gestures: {state.total_gestures_recognized} | FPS: {state.current_fps:.1f}"
    cv2.putText(ui_frame, stats_text, (30, CAM_HEIGHT + 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Draw buttons
    buttons = [
        ('clear', 'CLEAR'),
        ('speak', 'SPEAK'),
        ('pause_tts', 'PAUSE' if not state.tts_paused else 'RESUME'),
        ('settings', 'CONFIG')
    ]
    
    for btn_id, btn_text in buttons:
        rect = rects[btn_id]
        cv2.rectangle(ui_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), BUTTON_COLOR, -1)
        cv2.rectangle(ui_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), TEXT_COLOR, 1)
        text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = rect[0] + (rect[2] - text_size[0]) // 2
        text_y = rect[1] + (rect[3] + text_size[1]) // 2
        cv2.putText(ui_frame, btn_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
    # Text input area
    rect = rects['text_input']
    input_color = ACTIVE_COLOR if state.is_text_input_active else (255, 255, 255)
    cv2.rectangle(ui_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), input_color, -1)
    cv2.rectangle(ui_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), TEXT_COLOR, 2)
    
    display_text = state.input_text + ("|" if state.is_text_input_active else "")
    if not display_text.strip() and not state.is_text_input_active:
        display_text = "Click here to type text for sign display..."
    
    cv2.putText(ui_frame, display_text[:50], (rect[0] + 10, rect[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def draw_history_tab(ui_frame, rects):
    """Draw the history tab"""
    cv2.putText(ui_frame, "Recent Translations:", (30, CAM_HEIGHT + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    # Show last 5 translations
    y_offset = 100
    for i, entry in enumerate(reversed(state.translation_history[-5:])):
        if y_offset > CAM_HEIGHT + 200:
            break
        
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
        content = entry['content'][:40] + ('...' if len(entry['content']) > 40 else '')
        
        cv2.putText(ui_frame, f"{timestamp}: {content}", (30, CAM_HEIGHT + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        y_offset += 25

# ---------------- ENHANCED MAIN LOOP ----------------
def main():
    """Enhanced main application loop"""
    global state, config
    
    load_sign_images()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Enhanced window setup
    window_name = 'Enhanced Sign Language Translator'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT + TTS_DISPLAY_AREA_HEIGHT)
    cv2.setMouseCallback(window_name, enhanced_mouse_callback)
    
    logger.info("Application started successfully")
    
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # FPS calculation
            state.fps_counter += 1
            if time.time() - state.fps_start_time >= 1.0:
                state.current_fps = state.fps_counter
                state.fps_counter = 0
                state.fps_start_time = time.time()
            
            # Process image
            image = cv2.flip(image, 1)
            image_resized = cv2.resize(image, (CAM_WIDTH, CAM_HEIGHT))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Create enhanced UI frame
            ui_frame = np.full((WINDOW_HEIGHT + TTS_DISPLAY_AREA_HEIGHT, WINDOW_WIDTH, 3), 
                             UI_BG_COLOR, dtype=np.uint8)
            ui_frame[0:CAM_HEIGHT, 0:CAM_WIDTH] = image_resized
            
            # Process hand detection
            current_char = ""
            current_conf = 0.0
            top_predictions = {}
            hand_landmarks_list = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_landmarks_list = hand_landmarks.landmark
                prediction, confidence, top_predictions = recognize_gesture_enhanced(hand_landmarks_list)
                
                if confidence > config['confidence_threshold']:
                    state.prediction_history.append(prediction)
                    state.confidence_history.append(confidence)
                    current_char = f"{prediction.upper()} ({confidence*100:.1f}%)"
                    
                    # Enhanced stability logic
                    if len(state.prediction_history) >= config['prediction_window_size']:
                        most_common = Counter(state.prediction_history).most_common(1)
                        if most_common:
                            stable_prediction, count = most_common[0]
                            stability_ratio = count / len(state.prediction_history)
                            
                            if stability_ratio >= config['stability_threshold']:
                                if stable_prediction != state.last_stable_char:
                                    state.sentence.append(stable_prediction)
                                    state.last_stable_char = stable_prediction
                                    state.total_gestures_recognized += 1
                                    logger.info(f"Recognized: {stable_prediction} (confidence: {confidence:.2f})")
                else:
                    current_char = "Detecting..."
            else:
                if len(state.prediction_history) > 0:
                    state.prediction_history.clear()
                    state.confidence_history.clear()
                current_char = "No hand detected"
            
            # Draw enhanced hand landmarks
            landmark_canvas = draw_enhanced_landmarks(hand_landmarks_list)
            ui_frame[0:CAM_HEIGHT, CAM_WIDTH:CAM_WIDTH+LANDMARK_CANVAS_SIZE] = landmark_canvas
            
            # Draw enhanced UI
            draw_enhanced_ui(ui_frame, current_char, current_conf, top_predictions)
            
            # Handle text-to-sign display
            if state.text_to_sign_sentence and not state.tts_paused:
                current_time = time.time()
                if current_time - state.tts_last_update_time > config['tts_display_interval']:
                    state.tts_char_index = (state.tts_char_index + 1) % len(state.text_to_sign_sentence)
                    state.tts_last_update_time = current_time
                
                char_to_display = state.text_to_sign_sentence[state.tts_char_index].upper()
                if char_to_display in sign_images:
                    img_to_show = sign_images[char_to_display]
                    img_y_start = WINDOW_HEIGHT - 170
                    ui_frame[img_y_start:img_y_start+TTS_IMAGE_SIZE, 
                            CAM_WIDTH+180:CAM_WIDTH+180+TTS_IMAGE_SIZE] = img_to_show
                    
                    # Enhanced text display
                    cv2.putText(ui_frame, f"Showing: {char_to_display}", 
                               (CAM_WIDTH + 180, img_y_start + TTS_IMAGE_SIZE + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            
            cv2.imshow(window_name, ui_frame)
            
            # Enhanced keyboard input
            key = cv2.waitKey(5) & 0xFF
            if state.is_text_input_active:
                if key == 8:  # Backspace
                    state.input_text = state.input_text[:-1]
                elif key == 13:  # Enter
                    state.is_text_input_active = False
                    state.text_to_sign_sentence = ''.join(c for c in state.input_text.upper() if c.isalnum())
                    state.tts_char_index = 0
                    state.tts_last_update_time = time.time()
                    
                    # Add to history
                    if state.text_to_sign_sentence:
                        state.translation_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'text_to_sign',
                            'content': state.input_text,
                            'sign_sequence': state.text_to_sign_sentence
                        })
                        state.save_history()
                    
                    state.input_text = ""
                elif 32 <= key <= 126:  # Regular characters
                    state.input_text += chr(key)
            
            # Global hotkeys
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Quick save
                save_config()
                logger.info("Configuration saved")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    state.save_history()
    save_config()
    logger.info("Application closed successfully")

if __name__ == "__main__":
    main()