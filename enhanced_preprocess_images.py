import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_dir': "static_gesture_data",
    'output_csv': "processed_landmarks.csv",
    'metadata_file': "preprocessing_metadata.json",
    'min_confidence': 0.5,
    'image_quality_threshold': 0.7,
    'max_workers': 4,
    'augmentation': True,
    'normalize_landmarks': True,
    'extract_features': True,
    'validate_gestures': True
}

class EnhancedPreprocessor:
    def __init__(self, config=CONFIG):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1,
            min_detection_confidence=config['min_confidence'],
            min_tracking_confidence=config['min_confidence']
        )
        self.processed_count = 0
        self.failed_count = 0
        self.metadata = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'classes': {},
            'quality_stats': {},
            'processing_time': 0
        }

    def calculate_image_quality(self, image):
        """Calculate image quality metrics"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Normalize quality score (0-1)
        quality_score = min(1.0, (laplacian_var / 500.0 + contrast / 128.0 + 
                                 min(brightness, 255 - brightness) / 128.0) / 3.0)
        
        return {
            'sharpness': laplacian_var,
            'brightness': brightness,
            'contrast': contrast,
            'quality_score': quality_score
        }

    def extract_hand_features(self, landmarks):
        """Extract additional hand features beyond basic landmarks"""
        if not landmarks:
            return []
        
        # Convert to numpy array
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Basic landmarks (flattened)
        basic_features = coords.flatten()
        
        if not self.config['extract_features']:
            return basic_features.tolist()
        
        # Additional features
        features = list(basic_features)
        
        # 1. Finger angles and distances
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_mcp = [2, 5, 9, 13, 17]    # MCP joints
        
        # Calculate finger extension (tip to MCP distance)
        for tip, mcp in zip(finger_tips, finger_mcp):
            tip_pos = coords[tip][:2]  # x, y only
            mcp_pos = coords[mcp][:2]
            distance = np.linalg.norm(tip_pos - mcp_pos)
            features.append(distance)
        
        # 2. Hand span and palm size
        wrist = coords[0][:2]
        middle_mcp = coords[9][:2]
        palm_size = np.linalg.norm(middle_mcp - wrist)
        features.append(palm_size)
        
        # 3. Finger spread angles
        for i in range(len(finger_tips) - 1):
            tip1 = coords[finger_tips[i]][:2]
            tip2 = coords[finger_tips[i + 1]][:2]
            wrist_pos = coords[0][:2]
            
            # Calculate angle between fingers relative to wrist
            vec1 = tip1 - wrist_pos
            vec2 = tip2 - wrist_pos
            
            # Avoid division by zero
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
                features.append(angle)
            else:
                features.append(0.0)
        
        # 4. Hand orientation
        # Vector from wrist to middle finger MCP
        orientation_vec = middle_mcp - wrist
        orientation_angle = np.arctan2(orientation_vec[1], orientation_vec[0])
        features.append(orientation_angle)
        
        return features

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to make them scale and translation invariant"""
        if not landmarks:
            return landmarks
        
        # Convert to numpy array
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Center around wrist (landmark 0)
        wrist = coords[0]
        centered_coords = coords - wrist
        
        # Scale by palm size (distance from wrist to middle finger MCP)
        palm_size = np.linalg.norm(centered_coords[9][:2])  # Middle finger MCP
        if palm_size > 1e-6:
            centered_coords = centered_coords / palm_size
        
        # Update landmarks
        normalized_landmarks = []
        for i, coord in enumerate(centered_coords):
            lm = landmarks[i]
            # Create new landmark-like object
            normalized_lm = type('Landmark', (), {
                'x': coord[0], 'y': coord[1], 'z': coord[2]
            })()
            normalized_landmarks.append(normalized_lm)
        
        return normalized_landmarks

    def augment_landmarks(self, landmarks, augment_params=None):
        """Apply data augmentation to landmarks"""
        if not landmarks or not self.config['augmentation']:
            return [landmarks]
        
        augmented_versions = [landmarks]  # Include original
        
        if not augment_params:
            augment_params = {
                'rotation_angles': [-15, -7, 7, 15],  # degrees
                'scale_factors': [0.9, 1.1],
                'noise_std': 0.01
            }
        
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Rotation augmentation
        for angle in augment_params['rotation_angles']:
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated_coords = coords.copy()
            rotated_coords[:, :2] = np.dot(coords[:, :2], rotation_matrix.T)
            
            # Create new landmarks
            rotated_landmarks = []
            for i, coord in enumerate(rotated_coords):
                lm = type('Landmark', (), {
                    'x': coord[0], 'y': coord[1], 'z': coord[2]
                })()
                rotated_landmarks.append(lm)
            augmented_versions.append(rotated_landmarks)
        
        # Scale augmentation
        for scale in augment_params['scale_factors']:
            scaled_coords = coords * scale
            scaled_landmarks = []
            for i, coord in enumerate(scaled_coords):
                lm = type('Landmark', (), {
                    'x': coord[0], 'y': coord[1], 'z': coord[2]
                })()
                scaled_landmarks.append(lm)
            augmented_versions.append(scaled_landmarks)
        
        # Noise augmentation
        if augment_params['noise_std'] > 0:
            noise = np.random.normal(0, augment_params['noise_std'], coords.shape)
            noisy_coords = coords + noise
            noisy_landmarks = []
            for i, coord in enumerate(noisy_coords):
                lm = type('Landmark', (), {
                    'x': coord[0], 'y': coord[1], 'z': coord[2]
                })()
                noisy_landmarks.append(lm)
            augmented_versions.append(noisy_landmarks)
        
        return augmented_versions

    def validate_gesture(self, landmarks, label):
        """Validate if the detected landmarks represent a valid gesture"""
        if not landmarks or not self.config['validate_gestures']:
            return True
        
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Check if hand is reasonably sized and positioned
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        # Hand should occupy reasonable space in image
        if x_range < 0.1 or y_range < 0.1:
            return False
        
        # Check if landmarks are not clustered too tightly
        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        
        mean_distance = np.mean(distances)
        if mean_distance < 0.01:  # Too clustered
            return False
        
        return True

    def process_image(self, img_path, label):
        """Process a single image and extract features"""
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                return None
            
            # Calculate image quality
            quality_metrics = self.calculate_image_quality(img)
            if quality_metrics['quality_score'] < self.config['image_quality_threshold']:
                logger.debug(f"Low quality image skipped: {img_path}")
                return None
            
            # Process with MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                return None
            
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # Validate gesture
            if not self.validate_gesture(landmarks, label):
                return None
            
            # Normalize landmarks if requested
            if self.config['normalize_landmarks']:
                landmarks = self.normalize_landmarks(landmarks)
            
            # Extract features and apply augmentation
            augmented_landmarks_list = self.augment_landmarks(landmarks)
            
            processed_samples = []
            for aug_landmarks in augmented_landmarks_list:
                features = self.extract_hand_features(aug_landmarks)
                if features:  # Ensure we got valid features
                    sample_data = {
                        'features': features,
                        'label': label,
                        'source_file': str(img_path),
                        'quality_score': quality_metrics['quality_score'],
                        'augmented': len(processed_samples) > 0  # First is original
                    }
                    processed_samples.append(sample_data)
            
            return processed_samples
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            return None

    def process_class_directory(self, class_dir, label):
        """Process all images in a class directory with smart sampling for large datasets"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = []
        
        for img_path in Path(class_dir).iterdir():
            if img_path.suffix.lower() in image_extensions:
                image_paths.append(img_path)
        
        if not image_paths:
            logger.warning(f"No images found in {class_dir}")
            return []
        
        # Handle class imbalance by smart sampling
        max_samples_per_class = 500  # Limit to prevent memory issues with large classes
        if len(image_paths) > max_samples_per_class:
            logger.info(f"Class '{label}' has {len(image_paths)} images. Sampling {max_samples_per_class} for balanced dataset.")
            # Sort by file size to potentially get better quality images, then sample
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            # Take a mix: best quality + random sampling
            best_images = image_paths[:max_samples_per_class//3]
            remaining = image_paths[max_samples_per_class//3:]
            import random
            random.seed(42)  # For reproducibility
            random_images = random.sample(remaining, min(len(remaining), max_samples_per_class - len(best_images)))
            image_paths = best_images + random_images
        
        logger.info(f"Processing {len(image_paths)} images for class '{label}'")
        
        all_samples = []
        processed_count = 0
        failed_count = 0
        
        # Process images with progress bar
        with tqdm(image_paths, desc=f"Processing {label}") as pbar:
            for img_path in pbar:
                samples = self.process_image(img_path, label)
                if samples:
                    all_samples.extend(samples)
                    processed_count += 1
                else:
                    failed_count += 1
                
                pbar.set_postfix({
                    'processed': processed_count,
                    'failed': failed_count,
                    'samples': len(all_samples)
                })
        
        # Special handling for classes with very few samples (like 'space' with only 7 images)
        if label.lower() == 'space' and len(all_samples) < 20:
            logger.warning(f"Class '{label}' has very few samples ({len(all_samples)}). Consider collecting more data.")
            # Apply more aggressive augmentation for small classes
            if self.config['augmentation']:
                logger.info(f"Applying enhanced augmentation for small class '{label}'")
                enhanced_samples = []
                for sample in all_samples[:]:  # Copy to avoid modification during iteration
                    # Create more augmented versions for small classes
                    landmarks = self.reconstruct_landmarks_from_features(sample['features'])
                    if landmarks:
                        extra_augmented = self.augment_landmarks(landmarks, {
                            'rotation_angles': [-20, -15, -10, -5, 5, 10, 15, 20],
                            'scale_factors': [0.85, 0.9, 0.95, 1.05, 1.1, 1.15],
                            'noise_std': 0.005
                        })
                        
                        for aug_landmarks in extra_augmented[1:]:  # Skip original
                            features = self.extract_hand_features(aug_landmarks)
                            if features:
                                aug_sample = sample.copy()
                                aug_sample['features'] = features
                                aug_sample['augmented'] = True
                                enhanced_samples.append(aug_sample)
                
                all_samples.extend(enhanced_samples)
                logger.info(f"Enhanced '{label}' from {processed_count} to {len(all_samples)} samples")
        
        # Update metadata
        self.metadata['classes'][label] = {
            'total_images': len(image_paths),
            'original_image_count': len(list(Path(class_dir).glob('*.*'))),
            'processed_images': processed_count,
            'failed_images': failed_count,
            'total_samples': len(all_samples),
            'augmentation_ratio': len(all_samples) / max(1, processed_count)
        }
        
        return all_samples

    def reconstruct_landmarks_from_features(self, features):
        """Reconstruct landmarks from feature vector (basic version)"""
        try:
            # Basic features are first 63 elements (21 landmarks * 3 coordinates)
            basic_features = features[:63] if len(features) >= 63 else features
            if len(basic_features) < 63:
                return None
                
            landmarks = []
            for i in range(0, 63, 3):
                if i + 2 < len(basic_features):
                    lm = type('Landmark', (), {
                        'x': basic_features[i], 
                        'y': basic_features[i+1], 
                        'z': basic_features[i+2]
                    })()
                    landmarks.append(lm)
            
            return landmarks if len(landmarks) == 21 else None
        except:
            return None

    def process_dataset(self):
        """Process the entire dataset"""
        import time
        start_time = time.time()
        
        data_dir = Path(self.config['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Get all class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")
        
        logger.info(f"Found {len(class_dirs)} classes to process")
        
        all_samples = []
        
        # Process each class
        for class_dir in class_dirs:
            label = class_dir.name
            samples = self.process_class_directory(class_dir, label)
            all_samples.extend(samples)
        
        # Convert to DataFrame
        if not all_samples:
            raise ValueError("No samples were successfully processed")
        
        # Separate features and metadata
        features_data = []
        labels = []
        metadata_list = []
        
        for sample in all_samples:
            features_data.append(sample['features'])
            labels.append(sample['label'])
            metadata_list.append({
                'source_file': sample['source_file'],
                'quality_score': sample['quality_score'],
                'augmented': sample['augmented']
            })
        
        # Create DataFrame
        feature_columns = [f'feature_{i}' for i in range(len(features_data[0]))]
        df = pd.DataFrame(features_data, columns=feature_columns)
        df['label'] = labels
        
        # Save main dataset
        output_path = Path(self.config['output_csv'])
        df.to_csv(output_path, index=False)
        
        # Save detailed metadata
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df['label'] = labels
        metadata_df.to_csv(output_path.with_suffix('.metadata.csv'), index=False)
        
        # Update and save processing metadata
        processing_time = time.time() - start_time
        self.metadata.update({
            'total_images': sum(cls['total_images'] for cls in self.metadata['classes'].values()),
            'processed_images': sum(cls['processed_images'] for cls in self.metadata['classes'].values()),
            'failed_images': sum(cls['failed_images'] for cls in self.metadata['classes'].values()),
            'total_samples': len(all_samples),
            'processing_time': processing_time,
            'feature_count': len(feature_columns),
            'augmentation_enabled': self.config['augmentation'],
            'normalization_enabled': self.config['normalize_landmarks']
        })
        
        with open(self.config['metadata_file'], 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total classes: {len(self.metadata['classes'])}")
        logger.info(f"Total images processed: {self.metadata['processed_images']}")
        logger.info(f"Total samples generated: {self.metadata['total_samples']}")
        logger.info(f"Features per sample: {len(feature_columns)}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Metadata saved to: {self.config['metadata_file']}")
        
        # Class-wise summary
        logger.info("\nClass-wise Summary:")
        for label, stats in self.metadata['classes'].items():
            success_rate = (stats['processed_images'] / stats['total_images']) * 100
            logger.info(f"  {label}: {stats['processed_images']}/{stats['total_images']} "
                       f"({success_rate:.1f}%) -> {stats['total_samples']} samples")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sign Language Image Preprocessor')
    parser.add_argument('--data_dir', default=CONFIG['data_dir'], 
                       help='Input data directory')
    parser.add_argument('--output', default=CONFIG['output_csv'], 
                       help='Output CSV file')
    parser.add_argument('--no_augmentation', action='store_true', 
                       help='Disable data augmentation')
    parser.add_argument('--no_normalization', action='store_true', 
                       help='Disable landmark normalization')
    parser.add_argument('--min_confidence', type=float, default=CONFIG['min_confidence'],
                       help='Minimum detection confidence')
    parser.add_argument('--quality_threshold', type=float, default=CONFIG['image_quality_threshold'],
                       help='Minimum image quality threshold')
    parser.add_argument('--workers', type=int, default=CONFIG['max_workers'],
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Update configuration
    config = CONFIG.copy()
    config.update({
        'data_dir': args.data_dir,
        'output_csv': args.output,
        'augmentation': not args.no_augmentation,
        'normalize_landmarks': not args.no_normalization,
        'min_confidence': args.min_confidence,
        'image_quality_threshold': args.quality_threshold,
        'max_workers': args.workers
    })
    
    # Create and run preprocessor
    preprocessor = EnhancedPreprocessor(config)
    
    try:
        df = preprocessor.process_dataset()
        logger.info("✅ Preprocessing completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return False

if __name__ == "__main__":
    main()