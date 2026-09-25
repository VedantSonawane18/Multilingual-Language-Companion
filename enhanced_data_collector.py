import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    def __init__(self, config=None):
        self.config = config or {
            'data_path': 'enhanced_gesture_data',
            'image_data_path': 'static_gesture_data',
            'video_data_path': 'dynamic_gesture_data',
            'sequences_per_sign': 15,
            'images_per_sign': 30,
            'sequence_length': 30,
            'collection_fps': 10,
            'image_size': (640, 480),
            'save_images': True,
            'save_videos': False,
            'quality_threshold': 0.5,
            'auto_capture': False,
            'capture_delay': 3.0,
            'preview_time': 2.0
        }
        
        # Create directories
        for path_key in ['data_path', 'image_data_path', 'video_data_path']:
            Path(self.config[path_key]).mkdir(parents=True, exist_ok=True)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Collection state
        self.current_sign = ""
        self.collection_stats = defaultdict(int)
        self.session_start_time = datetime.now()
        
        # Quality assessment
        self.quality_metrics = []
        
        # Define comprehensive sign list
        self.define_sign_categories()
    
    def define_sign_categories(self):
        """Define comprehensive categories of signs to collect"""
        self.sign_categories = {
            'letters': [chr(ord('A') + i) for i in range(26)],
            'numbers': [str(i) for i in range(10)],
            'common_words': [
                'HELLO', 'GOODBYE', 'PLEASE', 'THANK_YOU', 'YES', 'NO',
                'HELP', 'STOP', 'GO', 'COME', 'SIT', 'STAND',
                'WATER', 'FOOD', 'MORE', 'FINISH', 'GOOD', 'BAD'
            ],
            'basic_actions': [
                'EAT', 'DRINK', 'SLEEP', 'WALK', 'RUN', 'JUMP',
                'READ', 'WRITE', 'LISTEN', 'LOOK', 'THINK', 'FEEL'
            ],
            'emotions': [
                'HAPPY', 'SAD', 'ANGRY', 'SURPRISED', 'AFRAID', 'LOVE'
            ],
            'special': ['SPACE', 'DELETE', 'ENTER', 'PAUSE']
        }
        
        # Flatten all signs
        self.all_signs = []
        for category, signs in self.sign_categories.items():
            self.all_signs.extend(signs)
        
        logger.info(f"Defined {len(self.all_signs)} signs across {len(self.sign_categories)} categories")
    
    def assess_hand_quality(self, image, landmarks):
        """Assess the quality of hand detection and pose"""
        if not landmarks:
            return 0.0, "No hand detected"
        
        # Convert to numpy array for easier processing
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        quality_factors = {}
        issues = []
        
        # 1. Hand visibility (all landmarks should be in frame)
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        margin = 0.05  # 5% margin from edges
        
        in_frame = np.all((x_coords > margin) & (x_coords < 1-margin) & 
                         (y_coords > margin) & (y_coords < 1-margin))
        quality_factors['visibility'] = 1.0 if in_frame else 0.5
        
        if not in_frame:
            issues.append("Hand near image boundary")
        
        # 2. Hand size (should occupy reasonable portion of image)
        hand_width = x_coords.max() - x_coords.min()
        hand_height = y_coords.max() - y_coords.min()
        hand_area = hand_width * hand_height
        
        ideal_area_range = (0.05, 0.4)  # 5% to 40% of image
        if hand_area < ideal_area_range[0]:
            quality_factors['size'] = 0.3
            issues.append("Hand too small")
        elif hand_area > ideal_area_range[1]:
            quality_factors['size'] = 0.7
            issues.append("Hand too large")
        else:
            size_score = 1.0 - abs(hand_area - 0.15) / 0.15  # Optimal around 15%
            quality_factors['size'] = max(0.5, size_score)
        
        # 3. Image sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate region of interest around hand
        x_min, x_max = int(x_coords.min() * image.shape[1]), int(x_coords.max() * image.shape[1])
        y_min, y_max = int(y_coords.min() * image.shape[0]), int(y_coords.max() * image.shape[0])
        
        # Add padding
        pad = 20
        x_min, x_max = max(0, x_min-pad), min(gray.shape[1], x_max+pad)
        y_min, y_max = max(0, y_min-pad), min(gray.shape[0], y_max+pad)
        
        hand_region = gray[y_min:y_max, x_min:x_max]
        
        if hand_region.size > 0:
            laplacian_var = cv2.Laplacian(hand_region, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 300.0)
            quality_factors['sharpness'] = sharpness_score
            
            if sharpness_score < 0.3:
                issues.append("Image blurry")
        else:
            quality_factors['sharpness'] = 0.5
        
        # 4. Lighting quality
        brightness = np.mean(hand_region) if hand_region.size > 0 else np.mean(gray)
        contrast = np.std(hand_region) if hand_region.size > 0 else np.std(gray)
        
        # Optimal brightness range
        brightness_score = 1.0 - abs(brightness - 128) / 128
        contrast_score = min(1.0, contrast / 50.0)
        
        quality_factors['lighting'] = (brightness_score + contrast_score) / 2
        
        if brightness < 50:
            issues.append("Too dark")
        elif brightness > 200:
            issues.append("Too bright")
        
        if contrast < 20:
            issues.append("Low contrast")
        
        # 5. Hand pose stability (compare with previous frames if available)
        quality_factors['stability'] = 1.0  # Placeholder for now
        
        # Calculate overall quality score
        weights = {
            'visibility': 0.3,
            'size': 0.25,
            'sharpness': 0.2,
            'lighting': 0.15,
            'stability': 0.1
        }
        
        overall_quality = sum(quality_factors[k] * weights[k] for k in weights.keys())
        
        # Create feedback message
        if overall_quality >= 0.8:
            feedback = "Excellent quality"
        elif overall_quality >= 0.6:
            feedback = "Good quality" + (f" ({', '.join(issues[:2])})" if issues else "")
        elif overall_quality >= 0.4:
            feedback = "Fair quality" + (f" ({', '.join(issues[:2])})" if issues else "")
        else:
            feedback = "Poor quality" + (f" ({', '.join(issues[:3])})" if issues else "")
        
        return overall_quality, feedback
    
    def draw_collection_interface(self, image, current_sign, progress_info, quality_score, feedback):
        """Draw an enhanced collection interface"""
        # Create info overlay
        overlay = image.copy()
        
        # Header bar
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Current sign info
        cv2.putText(image, f"Collecting: {current_sign}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Progress info
        progress_text = f"{progress_info['current']}/{progress_info['total']}"
        cv2.putText(image, progress_text, (image.shape[1] - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Quality indicator
        quality_color = (0, 255, 0) if quality_score >= 0.7 else (0, 255, 255) if quality_score >= 0.4 else (0, 0, 255)
        
        # Quality bar
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 50
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * quality_score), bar_y + bar_height), quality_color, -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Quality text
        cv2.putText(image, f"Quality: {quality_score:.1%}", (bar_x + bar_width + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1)
        
        # Footer with instructions and feedback
        footer_y = image.shape[0] - 60
        cv2.rectangle(image, (0, footer_y), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
        
        cv2.putText(image, feedback, (20, footer_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = "SPACE: Capture | S: Skip | Q: Quit | A: Auto-mode"
        cv2.putText(image, instructions, (20, footer_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return image
    
    def save_sample_image(self, image, landmarks, sign, sample_idx, quality_score):
        """Save a sample image with metadata"""
        # Create sign directory
        sign_dir = Path(self.config['image_data_path']) / sign
        sign_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp and quality
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quality_str = f"{quality_score:.2f}".replace('.', '_')
        filename = f"{sign}_{sample_idx:03d}_{timestamp}_q{quality_str}.jpg"
        filepath = sign_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), image)
        
        # Save landmark data
        landmarks_data = {
            'sign': sign,
            'sample_index': sample_idx,
            'timestamp': timestamp,
            'quality_score': quality_score,
            'landmarks': [[lm.x, lm.y, lm.z] for lm in landmarks] if landmarks else [],
            'image_path': str(filepath)
        }
        
        metadata_file = sign_dir / f"{sign}_{sample_idx:03d}_{timestamp}_landmarks.json"
        with open(metadata_file, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
        
        return filepath
    
    def save_session_stats(self):
        """Save collection session statistics"""
        stats = {
            'session_start': self.session_start_time.isoformat(),
            'session_end': datetime.now().isoformat(),
            'collection_stats': dict(self.collection_stats),
            'total_samples': sum(self.collection_stats.values()),
            'signs_completed': len(self.collection_stats),
            'average_quality': np.mean(self.quality_metrics) if self.quality_metrics else 0,
            'config': self.config
        }
        
        stats_file = Path(self.config['data_path']) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Session statistics saved to {stats_file}")
        return stats
    
    def collect_sign_images(self, sign):
        """Collect images for a specific sign"""
        logger.info(f"Starting collection for sign: {sign}")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['image_size'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['image_size'][1])
        
        collected_count = 0
        target_count = self.config['images_per_sign']
        
        # Auto-capture settings
        auto_mode = False
        auto_capture_timer = 0
        last_auto_capture = 0
        
        # Quality tracking
        recent_qualities = []
        
        logger.info(f"Collect {target_count} images for '{sign}'. Press SPACE to capture, 'a' for auto-mode, 'q' to quit.")
        
        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extract landmarks and assess quality
            landmarks = None
            quality_score = 0.0
            feedback = "No hand detected"
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                quality_score, feedback = self.assess_hand_quality(frame, landmarks)
                recent_qualities.append(quality_score)
                
                # Keep only recent qualities
                if len(recent_qualities) > 30:
                    recent_qualities = recent_qualities[-30:]
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            
            # Draw collection interface
            progress_info = {'current': collected_count, 'total': target_count}
            frame = self.draw_collection_interface(frame, sign, progress_info, quality_score, feedback)
            
            # Auto-capture logic
            current_time = time.time()
            if auto_mode and landmarks and quality_score >= self.config['quality_threshold']:
                if current_time - last_auto_capture >= self.config['capture_delay']:
                    # Check if quality has been stable
                    if len(recent_qualities) >= 5 and np.mean(recent_qualities[-5:]) >= self.config['quality_threshold']:
                        # Auto capture
                        filepath = self.save_sample_image(frame, landmarks, sign, collected_count + 1, quality_score)
                        collected_count += 1
                        last_auto_capture = current_time
                        self.quality_metrics.append(quality_score)
                        
                        logger.info(f"Auto-captured {collected_count}/{target_count} (Quality: {quality_score:.2f})")
                        
                        # Flash effect
                        flash_frame = frame.copy()
                        cv2.rectangle(flash_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 10)
                        cv2.imshow('Sign Language Data Collection', flash_frame)
                        cv2.waitKey(100)
                else:
                    # Show countdown
                    remaining = self.config['capture_delay'] - (current_time - last_auto_capture)
                    cv2.putText(frame, f"Auto-capture in: {remaining:.1f}s", 
                               (frame.shape[1] - 250, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show auto-mode indicator
            if auto_mode:
                cv2.putText(frame, "AUTO MODE", (frame.shape[1] - 150, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and not auto_mode:  # Manual capture
                if landmarks and quality_score >= self.config['quality_threshold']:
                    filepath = self.save_sample_image(frame, landmarks, sign, collected_count + 1, quality_score)
                    collected_count += 1
                    self.quality_metrics.append(quality_score)
                    logger.info(f"Captured {collected_count}/{target_count} (Quality: {quality_score:.2f})")
                else:
                    logger.warning(f"Quality too low ({quality_score:.2f}) or no hand detected")
            
            elif key == ord('a'):  # Toggle auto mode
                auto_mode = not auto_mode
                last_auto_capture = current_time
                logger.info(f"Auto-capture mode: {'ON' if auto_mode else 'OFF'}")
            
            elif key == ord('s'):  # Skip current sign
                logger.info(f"Skipping sign '{sign}' (collected {collected_count}/{target_count})")
                break
            
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return collected_count
        
        cap.release()
        self.collection_stats[sign] = collected_count
        logger.info(f"Completed collection for '{sign}': {collected_count} images")
        
        return collected_count
    
    def collect_all_signs(self, categories=None, custom_signs=None):
        """Collect data for all signs or specified categories"""
        if custom_signs:
            signs_to_collect = custom_signs
        elif categories:
            signs_to_collect = []
            for category in categories:
                if category in self.sign_categories:
                    signs_to_collect.extend(self.sign_categories[category])
                else:
                    logger.warning(f"Unknown category: {category}")
        else:
            signs_to_collect = self.all_signs.copy()
        
        logger.info(f"Starting data collection for {len(signs_to_collect)} signs")
        
        total_collected = 0
        successful_signs = []
        
        try:
            for i, sign in enumerate(signs_to_collect):
                logger.info(f"Progress: {i+1}/{len(signs_to_collect)} - {sign}")
                
                # Show preview/preparation time
                self.show_sign_preview(sign)
                
                # Collect images for this sign
                collected = self.collect_sign_images(sign)
                total_collected += collected
                
                if collected > 0:
                    successful_signs.append(sign)
                
                # Brief pause between signs
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            
            # Save session statistics
            stats = self.save_session_stats()
            
            # Print summary
            logger.info("=" * 60)
            logger.info("DATA COLLECTION SESSION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Signs attempted: {len(signs_to_collect)}")
            logger.info(f"Signs completed: {len(successful_signs)}")
            logger.info(f"Total images collected: {total_collected}")
            
            if self.quality_metrics:
                avg_quality = np.mean(self.quality_metrics)
                logger.info(f"Average quality score: {avg_quality:.3f}")
            
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
            logger.info(f"Session duration: {session_duration/60:.1f} minutes")
            
            # Detailed breakdown
            logger.info("\nCollection breakdown:")
            for sign in successful_signs:
                count = self.collection_stats[sign]
                logger.info(f"  {sign}: {count} images")
        
        return total_collected
    
    def show_sign_preview(self, sign):
        """Show a preview screen for the upcoming sign"""
        # Create preview frame
        preview_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add sign information
        cv2.putText(preview_frame, "GET READY", (200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.putText(preview_frame, f"Next Sign: {sign}", (150, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        
        # Add instructions
        instructions = [
            "Position your hand clearly in view",
            "Ensure good lighting",
            "Make the sign steadily",
            f"We'll collect {self.config['images_per_sign']} images"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(preview_frame, instruction, (50, 320 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show countdown
        for countdown in range(int(self.config['preview_time']), 0, -1):
            frame_copy = preview_frame.copy()
            cv2.putText(frame_copy, f"Starting in {countdown}...", (200, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Data Collection', frame_copy)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    
    def validate_collected_data(self):
        """Validate the collected data and provide quality report"""
        logger.info("Validating collected data...")
        
        validation_report = {
            'total_signs': 0,
            'total_images': 0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'sign_details': {},
            'issues': []
        }
        
        image_data_path = Path(self.config['image_data_path'])
        
        for sign_dir in image_data_path.iterdir():
            if not sign_dir.is_dir():
                continue
            
            sign = sign_dir.name
            images = list(sign_dir.glob('*.jpg')) + list(sign_dir.glob('*.png'))
            metadata_files = list(sign_dir.glob('*_landmarks.json'))
            
            validation_report['total_signs'] += 1
            validation_report['total_images'] += len(images)
            
            sign_quality_scores = []
            
            # Analyze metadata files
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        quality = data.get('quality_score', 0)
                        sign_quality_scores.append(quality)
                        
                        if quality >= 0.7:
                            validation_report['quality_distribution']['high'] += 1
                        elif quality >= 0.4:
                            validation_report['quality_distribution']['medium'] += 1
                        else:
                            validation_report['quality_distribution']['low'] += 1
                except:
                    continue
            
            # Sign-specific analysis
            avg_quality = np.mean(sign_quality_scores) if sign_quality_scores else 0
            validation_report['sign_details'][sign] = {
                'image_count': len(images),
                'metadata_count': len(metadata_files),
                'average_quality': avg_quality,
                'quality_scores': sign_quality_scores
            }
            
            # Check for issues
            if len(images) != len(metadata_files):
                validation_report['issues'].append(f"{sign}: Mismatch between images and metadata")
            
            if len(images) < self.config['images_per_sign'] * 0.5:
                validation_report['issues'].append(f"{sign}: Too few images ({len(images)})")
            
            if avg_quality < 0.4:
                validation_report['issues'].append(f"{sign}: Low average quality ({avg_quality:.2f})")
        
        # Save validation report
        report_file = Path(self.config['data_path']) / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Print summary
        logger.info("VALIDATION REPORT")
        logger.info("=" * 40)
        logger.info(f"Total signs: {validation_report['total_signs']}")
        logger.info(f"Total images: {validation_report['total_images']}")
        logger.info("Quality distribution:")
        logger.info(f"  High (≥70%): {validation_report['quality_distribution']['high']}")
        logger.info(f"  Medium (40-69%): {validation_report['quality_distribution']['medium']}")
        logger.info(f"  Low (<40%): {validation_report['quality_distribution']['low']}")
        
        if validation_report['issues']:
            logger.warning(f"Found {len(validation_report['issues'])} issues:")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ No issues found!")
        
        return validation_report

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sign Language Data Collector')
    parser.add_argument('--data_path', default='enhanced_gesture_data',
                       help='Base data directory')
    parser.add_argument('--images_per_sign', type=int, default=30,
                       help='Number of images to collect per sign')
    parser.add_argument('--categories', nargs='+', 
                       choices=['letters', 'numbers', 'common_words', 'basic_actions', 'emotions', 'special'],
                       help='Sign categories to collect')
    parser.add_argument('--custom_signs', nargs='+',
                       help='Custom list of signs to collect')
    parser.add_argument('--auto_mode', action='store_true',
                       help='Start in auto-capture mode')
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                       help='Minimum quality threshold for auto-capture')
    parser.add_argument('--capture_delay', type=float, default=3.0,
                       help='Delay between auto-captures (seconds)')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate existing data')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_path': args.data_path,
        'image_data_path': os.path.join(args.data_path, 'static_gesture_data'),
        'video_data_path': os.path.join(args.data_path, 'dynamic_gesture_data'),
        'sequences_per_sign': 15,
        'images_per_sign': args.images_per_sign,
        'sequence_length': 30,
        'collection_fps': 10,
        'image_size': (640, 480),
        'save_images': True,
        'save_videos': False,
        'quality_threshold': args.quality_threshold,
        'auto_capture': args.auto_mode,
        'capture_delay': args.capture_delay,
        'preview_time': 2.0
    }
    
    # Create collector
    collector = EnhancedDataCollector(config)
    
    if args.validate_only:
        # Just validate existing data
        collector.validate_collected_data()
        return
    
    try:
        # Collect data
        total_collected = collector.collect_all_signs(
            categories=args.categories,
            custom_signs=args.custom_signs
        )
        
        # Validate collected data
        if total_collected > 0:
            logger.info("Validating collected data...")
            collector.validate_collected_data()
        
        logger.info("✅ Data collection completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()