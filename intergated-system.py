"""
COMPLETE AMBIENT WILDLIFE MONITORING AGENT
With Intelligent Frame Selection & All 6 Phases

Pipeline:
1. Motion Detection (MOG2) ✓ WORKING
2. Keyframe Sampling → Extract frames from motion regions with saving ✓
3. Intelligent Frame Selection ✓ NEW - reduce to most interesting frames
4. Classification → Predict animal class (only on selected frames)
5. VLM Verification → Validate predictions
6. Data Drift Detection → Compare classifier vs VLM
7. Autonomous Retraining Recommendation → Autonomous decision

This is the competition-winning ambient agent!
"""

import cv2
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
import json

# ============================================================================
# PHASE 1: ROBUST MOTION DETECTION (MOG2)
# ============================================================================

class MotionDetector:
    """MOG2 Background Subtraction - optimized for camera traps"""
    
    def __init__(self, history=500, var_threshold=16):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            history=history,
            varThreshold=var_threshold
        )
    
    def detect_motion_regions(self, video_path, motion_threshold=0.02):
        """Detect motion regions in video"""
        print(f"\n{'='*60}")
        print(f"PHASE 1: Motion Detection (MOG2)")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        motion_frames = []
        frame_idx = 0
        
        print(f"Video: {total_frames} frames @ {fps} FPS")
        print(f"Motion threshold: {motion_threshold*100:.1f}% of image")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate motion intensity
            motion_intensity = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
            
            if motion_intensity > motion_threshold:
                motion_frames.append((frame_idx, motion_intensity))
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames}...")
        
        cap.release()
        
        # Group into regions
        motion_regions = self._group_motion_frames(motion_frames, gap_threshold=5)
        
        print(f"✓ Motion frames: {len(motion_frames)} ({len(motion_frames)/total_frames*100:.1f}%)")
        print(f"✓ Motion regions: {len(motion_regions)}")
        
        for i, (start, end, intensity) in enumerate(motion_regions):
            duration = (end - start) / fps
            print(f"  Region {i+1}: frames {start}-{end} ({duration:.1f}s, intensity: {intensity:.3f})")
        
        return motion_regions
    
    @staticmethod
    def _group_motion_frames(motion_frames, gap_threshold=5):
        """Group consecutive motion frames into regions"""
        if not motion_frames:
            return []
        
        regions = []
        region_start = motion_frames[0][0]
        region_max_intensity = motion_frames[0][1]
        
        for i in range(1, len(motion_frames)):
            current_frame = motion_frames[i][0]
            current_intensity = motion_frames[i][1]
            prev_frame = motion_frames[i-1][0]
            
            region_max_intensity = max(region_max_intensity, current_intensity)
            
            if current_frame - prev_frame > gap_threshold:
                regions.append((region_start, prev_frame, region_max_intensity))
                region_start = current_frame
                region_max_intensity = current_intensity
        
        regions.append((region_start, motion_frames[-1][0], region_max_intensity))
        
        return regions


# ============================================================================
# PHASE 2: KEYFRAME SAMPLING
# ============================================================================

class KeyframeSampler:
    """Extract and save representative keyframes from motion regions"""
    
    @staticmethod
    def sample_keyframes(video_path, motion_regions, samples_per_region=3, save_dir=None):
        """Sample keyframes with highest motion intensity from each region"""
        print(f"\n{'='*60}")
        print(f"PHASE 2: Keyframe Sampling")
        print(f"{'='*60}")
        
        keyframes = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving keyframes to: {save_dir}")
        
        for region_idx, (start_frame, end_frame, region_intensity) in enumerate(motion_regions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            region_frames = []
            prev_gray = None
            
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion_intensity = np.mean(diff)
                else:
                    motion_intensity = 0
                
                region_frames.append({
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'motion_intensity': motion_intensity
                })
                
                prev_gray = gray
            
            # Sort by motion and take top samples
            region_frames.sort(key=lambda x: x['motion_intensity'], reverse=True)
            top_frames = region_frames[:samples_per_region]
            
            for sample_idx, frame_data in enumerate(top_frames):
                timestamp = frame_data['frame_idx'] / fps
                
                # Save keyframe if directory specified
                saved_path = None
                if save_dir:
                    filename = f"region_{region_idx:02d}_sample_{sample_idx:02d}_frame_{frame_data['frame_idx']:04d}_ts_{timestamp:.2f}s.jpg"
                    saved_path = os.path.join(save_dir, filename)
                    cv2.imwrite(saved_path, frame_data['frame'])
                
                keyframes.append({
                    'region_id': region_idx,
                    'frame_idx': frame_data['frame_idx'],
                    'timestamp': timestamp,
                    'frame': frame_data['frame'],
                    'motion_intensity': frame_data['motion_intensity'],
                    'saved_path': saved_path
                })
            
            print(f"Region {region_idx+1}: Sampled {len(top_frames)} keyframes")
        
        cap.release()
        
        print(f"✓ Total keyframes: {len(keyframes)}")
        if save_dir:
            print(f"✓ Keyframes saved to: {save_dir}")
        
        return keyframes


# ============================================================================
# PHASE 2.5: INTELLIGENT FRAME SELECTION
# ============================================================================

class FrameSelector:
    """Intelligently select interesting keyframes before classification"""
    
    @staticmethod
    def calculate_sharpness(frame):
        """Calculate frame sharpness using Laplacian variance"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def calculate_entropy(frame):
        """Calculate frame entropy (information content)"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    @staticmethod
    def calculate_contrast(frame):
        """Calculate frame contrast"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.std()
    
    @staticmethod
    def select_diverse_frames(keyframes, num_select=5, method='motion_intensity'):
        """Select diverse keyframes using different strategies"""
        
        if len(keyframes) <= num_select:
            return keyframes
        
        print(f"\n{'='*60}")
        print(f"PHASE 2.5: Intelligent Frame Selection")
        print(f"{'='*60}")
        print(f"Total keyframes: {len(keyframes)}")
        print(f"Selecting: {num_select} frames (method: {method})")
        
        # Calculate metrics
        metrics = []
        for i, kf in enumerate(keyframes):
            frame = kf['frame']
            metrics.append({
                'index': i,
                'keyframe': kf,
                'sharpness': FrameSelector.calculate_sharpness(frame),
                'entropy': FrameSelector.calculate_entropy(frame),
                'contrast': FrameSelector.calculate_contrast(frame),
                'motion_intensity': kf['motion_intensity']
            })
        
        # Normalize metrics
        sharpness_scores = np.array([m['sharpness'] for m in metrics])
        entropy_scores = np.array([m['entropy'] for m in metrics])
        contrast_scores = np.array([m['contrast'] for m in metrics])
        motion_scores = np.array([m['motion_intensity'] for m in metrics])
        
        if sharpness_scores.max() > 0:
            sharpness_scores = sharpness_scores / sharpness_scores.max()
        if entropy_scores.max() > 0:
            entropy_scores = entropy_scores / entropy_scores.max()
        if contrast_scores.max() > 0:
            contrast_scores = contrast_scores / contrast_scores.max()
        if motion_scores.max() > 0:
            motion_scores = motion_scores / motion_scores.max()
        
        # Score based on method
        if method == 'motion_intensity':
            scores = motion_scores
        elif method == 'quality':
            scores = (sharpness_scores * 0.4 + entropy_scores * 0.3 + contrast_scores * 0.3)
        elif method == 'entropy':
            scores = entropy_scores
        elif method == 'contrast':
            scores = contrast_scores
        else:  # balanced
            scores = (motion_scores * 0.3 + sharpness_scores * 0.3 + 
                     entropy_scores * 0.2 + contrast_scores * 0.2)
        
        for i, metric in enumerate(metrics):
            metric['score'] = scores[i]
        
        # Sort and select
        metrics.sort(key=lambda x: x['score'], reverse=True)
        selected = metrics[:num_select]
        selected.sort(key=lambda x: (x['keyframe']['region_id'], x['keyframe']['frame_idx']))
        
        print(f"\nSelected frames:")
        print(f"{'Idx':>3} {'Region':>6} {'Frame':>6} {'Motion':>7} {'Sharp':>7} {'Score':>7}")
        print("-" * 50)
        for m in selected:
            kf = m['keyframe']
            print(f"{m['index']:3d} {kf['region_id']:6d} {kf['frame_idx']:6d} "
                  f"{m['motion_intensity']:7.3f} {m['sharpness']:7.1f} {m['score']:7.3f}")
        
        result = [m['keyframe'] for m in selected]
        print(f"\n✓ Selected {len(result)} frames ({100*(len(keyframes)-len(result))/len(keyframes):.0f}% reduction)")
        
        return result
    
    @staticmethod
    def select_by_region(keyframes, max_per_region=1):
        """Select best frame(s) from each motion region"""
        
        print(f"\n{'='*60}")
        print(f"PHASE 2.5: Intelligent Frame Selection (By Region)")
        print(f"{'='*60}")
        print(f"Total keyframes: {len(keyframes)}")
        print(f"Selecting: max {max_per_region} frame(s) per region")
        
        regions = defaultdict(list)
        for kf in keyframes:
            regions[kf['region_id']].append(kf)
        
        selected = []
        for region_id in sorted(regions.keys()):
            region_frames = sorted(regions[region_id], 
                                 key=lambda x: x['motion_intensity'], reverse=True)
            top = region_frames[:max_per_region]
            selected.extend(top)
            print(f"Region {region_id:2d}: Selected {len(top)} frame(s) "
                  f"(motion: {top[0]['motion_intensity']:.3f})")
        
        print(f"\n✓ Selected {len(selected)} frames ({100*(len(keyframes)-len(selected))/len(keyframes):.0f}% reduction)")
        
        return selected


# ============================================================================
# PHASE 3: CLASSIFICATION
# ============================================================================

class FrameClassifier:
    """Classify sampled keyframes and save results"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Handle both dict and list formats for class names
        if hasattr(self.model, 'names'):
            if isinstance(self.model.names, dict):
                # Convert dict to list of values (preserving order)
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self.model.names
        else:
            self.class_names = ['background', 'fox', 'coati', 'ocelot', 'deer']
    
    def classify_keyframes(self, keyframes, save_dir=None):
        """Classify each keyframe and optionally save to disk"""
        print(f"\n{'='*60}")
        print(f"PHASE 3: Classification")
        print(f"{'='*60}")
        
        # Create save directories if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Create subdirectories for each class
            for class_name in self.class_names:
                class_dir = os.path.join(save_dir, str(class_name).replace(' ', '_'))
                os.makedirs(class_dir, exist_ok=True)
            print(f"Saving classified images to: {save_dir}")
        
        classifications = []
        
        for keyframe_data in keyframes:
            frame = keyframe_data['frame']
            
            # Predict
            results = self.model.predict(frame, verbose=False)
            pred = results[0]
            
            probs = pred.probs.data.cpu().numpy()
            predicted_class_id = np.argmax(probs)
            confidence = probs[predicted_class_id]
            
            if predicted_class_id < len(self.class_names):
                predicted_class = str(self.class_names[predicted_class_id])
            else:
                predicted_class = f"class_{predicted_class_id}"
            
            # Save image if directory specified
            saved_path = None
            if save_dir:
                class_dir = os.path.join(save_dir, predicted_class.replace(' ', '_'))
                filename = (f"region_{keyframe_data['region_id']:02d}_frame_{keyframe_data['frame_idx']:04d}_"
                           f"ts_{keyframe_data['timestamp']:.2f}s_class_{predicted_class.replace(' ', '_')}_"
                           f"conf_{confidence:.3f}.jpg")
                saved_path = os.path.join(class_dir, filename)
                cv2.imwrite(saved_path, frame)
            
            classification = {
                'frame_idx': keyframe_data['frame_idx'],
                'timestamp': keyframe_data['timestamp'],
                'region_id': keyframe_data['region_id'],
                'predicted_class': predicted_class,
                'class_id': int(predicted_class_id),
                'confidence': float(confidence),
                'all_probs': {str(self.class_names[i]) if i < len(self.class_names) else f'class_{i}': 
                             float(probs[i]) for i in range(len(probs))},
                'frame': frame,
                'saved_path': saved_path
            }
            
            classifications.append(classification)
            
            print(f"Frame {keyframe_data['frame_idx']:4d} @ {keyframe_data['timestamp']:6.2f}s → "
                  f"{predicted_class:15s} (conf: {confidence:.3f})")
        
        if save_dir:
            print(f"✓ Classified images saved to: {save_dir}")
        
        return classifications


# ============================================================================
# PHASE 4: VLM VERIFICATION
# ============================================================================

class VLMVerifier:
    """Verify classifier predictions with VLM"""
    
    def verify_predictions(self, classifications):
        """Use VLM to validate classifier predictions"""
        print(f"\n{'='*60}")
        print(f"PHASE 4: VLM Verification")
        print(f"{'='*60}")
        
        verified_results = []
        
        for clf in classifications:
            vlm_assessment = self._mock_vlm_assessment(clf)
            
            classifier_says = clf['predicted_class']
            vlm_says = vlm_assessment['detected_object']
            
            match = (classifier_says.lower() == vlm_says.lower()) or \
                    (vlm_says == 'unknown' and classifier_says.lower() == 'background')
            
            confidence_agreement = abs(clf['confidence'] - vlm_assessment['confidence'])
            
            result = {
                'frame_idx': clf['frame_idx'],
                'timestamp': clf['timestamp'],
                'classifier_prediction': classifier_says,
                'classifier_confidence': clf['confidence'],
                'vlm_assessment': vlm_says,
                'vlm_confidence': vlm_assessment['confidence'],
                'match': match,
                'confidence_agreement': float(confidence_agreement),
                'all_probs': clf['all_probs'],
                'vlm_reasoning': vlm_assessment['reasoning'],
                'verification_status': 'verified' if match else 'mismatch'
            }
            
            verified_results.append(result)
            
            status_symbol = "✓" if match else "✗"
            print(f"{status_symbol} Frame {clf['frame_idx']}: {classifier_says:15s} "
                  f"({clf['confidence']:.3f}) vs {vlm_says:15s} ({vlm_assessment['confidence']:.3f})")
        
        return verified_results
    
    @staticmethod
    def _mock_vlm_assessment(classification):
        """Mock VLM response (in production, use MobileVLM-3B)"""
        classifier_conf = classification['confidence']
        
        if classifier_conf > 0.95:
            return {
                'detected_object': classification['predicted_class'],
                'confidence': classifier_conf * 0.95,
                'reasoning': f"Clearly contains {classification['predicted_class']}"
            }
        elif classifier_conf > 0.7:
            return {
                'detected_object': classification['predicted_class'],
                'confidence': classifier_conf * 0.85,
                'reasoning': f"Likely {classification['predicted_class']}"
            }
        elif classifier_conf > 0.5:
            return {
                'detected_object': 'unknown',
                'confidence': 0.4,
                'reasoning': "Ambiguous - uncertain prediction"
            }
        else:
            return {
                'detected_object': 'background',
                'confidence': 0.3,
                'reasoning': "Too uncertain to classify"
            }


# ============================================================================
# PHASE 5: DATA DRIFT DETECTION
# ============================================================================

class DataDriftDetector:
    """Detect data drift from verification mismatches"""
    
    def __init__(self, mismatch_threshold=0.3):
        self.mismatch_threshold = mismatch_threshold
    
    def detect_drift(self, verified_results):
        """Analyze results to detect drift"""
        print(f"\n{'='*60}")
        print(f"PHASE 5: Data Drift Detection")
        print(f"{'='*60}")
        
        if not verified_results:
            return {'drift_detected': False, 'drift_score': 0.0}
        
        mismatches = sum(1 for r in verified_results if not r['match'])
        mismatch_rate = mismatches / len(verified_results)
        
        conf_disagreements = [r['confidence_agreement'] for r in verified_results]
        avg_conf_disagreement = np.mean(conf_disagreements)
        
        predictions = defaultdict(int)
        for r in verified_results:
            predictions[r['classifier_prediction']] += 1
        
        total_preds = len(verified_results)
        max_class_count = max(predictions.values()) if predictions else 0
        class_dominance = max_class_count / total_preds if total_preds > 0 else 0
        
        drift_score = (
            mismatch_rate * 0.4 +
            avg_conf_disagreement * 0.3 +
            (class_dominance - 0.7) * 0.3 if class_dominance > 0.7 else 0
        )
        
        drift_detected = (mismatch_rate > self.mismatch_threshold or 
                         class_dominance > 0.9)
        
        print(f"Mismatch rate: {mismatch_rate:.1%} (threshold: {self.mismatch_threshold:.1%})")
        print(f"Avg confidence disagreement: {avg_conf_disagreement:.3f}")
        print(f"Class dominance: {class_dominance:.1%}")
        print(f"Overall drift score: {drift_score:.3f}")
        
        print(f"\nPrediction distribution:")
        for class_name, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name:20s}: {count:3d} ({count/total_preds*100:5.1f}%)")
        
        status = "⚠️  DRIFT DETECTED" if drift_detected else "✓ NO DRIFT"
        print(f"\n{status}")
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'mismatch_rate': mismatch_rate,
            'confidence_disagreement': avg_conf_disagreement,
            'class_dominance': class_dominance,
            'predictions': dict(predictions)
        }


# ============================================================================
# PHASE 6: AUTONOMOUS RETRAINING RECOMMENDATION
# ============================================================================

class RetrainingRecommender:
    """Make autonomous retraining decisions"""
    
    def __init__(self):
        self.history = []
    
    def recommend_action(self, drift_analysis, verified_results):
        """Make autonomous recommendation"""
        print(f"\n{'='*60}")
        print(f"PHASE 6: Autonomous Retraining Recommendation")
        print(f"{'='*60}")
        
        drift_score = drift_analysis['drift_score']
        mismatch_rate = drift_analysis['mismatch_rate']
        drift_detected = drift_analysis['drift_detected']
        
        if drift_detected:
            if mismatch_rate > 0.5:
                recommendation = 'URGENT_RETRAIN'
                confidence = 0.95
                rationale = (
                    f"High mismatch rate ({mismatch_rate:.1%}). "
                    "Model predictions significantly diverge from VLM assessment."
                )
            else:
                recommendation = 'RETRAIN_RECOMMENDED'
                confidence = 0.75
                rationale = (
                    f"Moderate mismatch rate ({mismatch_rate:.1%}). "
                    "Consider retraining with newly verified samples."
                )
        else:
            if drift_score < 0.1:
                recommendation = 'CONTINUE_MONITORING'
                confidence = 0.9
                rationale = "Model performing well. Continue monitoring."
            else:
                recommendation = 'MONITOR_CLOSELY'
                confidence = 0.7
                rationale = f"Drift score {drift_score:.3f}. Monitor for changes."
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale,
            'drift_score': drift_score,
            'mismatch_rate': mismatch_rate,
            'verified_samples': len(verified_results)
        }
        
        self.history.append(decision)
        return decision
    
    def print_recommendation(self, decision):
        """Pretty print the recommendation"""
        print(f"\n{'='*60}")
        print(f"AMBIENT AGENT AUTONOMOUS DECISION")
        print(f"{'='*60}")
        print(f"Recommendation: {decision['recommendation']}")
        print(f"Confidence: {decision['confidence']:.1%}")
        print(f"Rationale: {decision['rationale']}")
        print(f"Verified samples: {decision['verified_samples']}")
        
        if decision['recommendation'].startswith('RETRAIN'):
            print(f"\n⚠️  ACTION REQUIRED:")
            print(f"   - Collect verified samples ({decision['verified_samples']} predictions)")
            print(f"   - Review mismatches for new classes")
            print(f"   - Retrain model on combined dataset")
        else:
            print(f"\n✓ System operating normally")


# ============================================================================
# MAIN: COMPLETE AMBIENT AGENT PIPELINE
# ============================================================================

def ambient_monitoring_agent(video_path, classifier_model_path, 
                            save_keyframes_dir=None,
                            save_classified_dir=None,
                            frame_selection_method='by_region',
                            num_select=5):
    """
    Complete ambient monitoring agent with intelligent frame selection.
    
    Args:
        video_path: Path to video file
        classifier_model_path: Path to YOLO classifier model
        save_keyframes_dir: Directory to save sampled keyframes (optional)
        save_classified_dir: Directory to save classified images organized by class (optional)
        frame_selection_method: 'by_region', 'balanced', 'motion_intensity', 'quality'
        num_select: Number of frames to select (only for non-'by_region' methods)
    """
    print("\n" + "="*70)
    print("AMBIENT WILDLIFE MONITORING AGENT")
    print("Complete Pipeline: Motion → Sampling → Selection → Classification → Verification → Drift Detection")
    print("="*70)
    
    # Initialize components
    motion_detector = MotionDetector()
    sampler = KeyframeSampler()
    frame_selector = FrameSelector()
    classifier = FrameClassifier(classifier_model_path)
    vlm_verifier = VLMVerifier()
    drift_detector = DataDriftDetector(mismatch_threshold=0.3)
    recommender = RetrainingRecommender()
    
    # PHASE 1: Motion Detection
    motion_regions = motion_detector.detect_motion_regions(video_path)
    
    if not motion_regions:
        print("\n✗ No motion detected. Exiting.")
        return None
    
    # PHASE 2: Keyframe Sampling
    keyframes = sampler.sample_keyframes(video_path, motion_regions, 
                                        samples_per_region=3,
                                        save_dir=save_keyframes_dir)
    
    # PHASE 2.5: Intelligent Frame Selection
    if frame_selection_method == 'by_region':
        selected_frames = frame_selector.select_by_region(keyframes, max_per_region=1)
    else:
        selected_frames = frame_selector.select_diverse_frames(
            keyframes, num_select=num_select, method=frame_selection_method
        )
    
    # PHASE 3: Classification (only on selected frames)
    classifications = classifier.classify_keyframes(selected_frames, save_dir=save_classified_dir)
    
    # PHASE 4: VLM Verification
    verified_results = vlm_verifier.verify_predictions(classifications)
    
    # PHASE 5: Data Drift Detection
    drift_analysis = drift_detector.detect_drift(verified_results)
    
    # PHASE 6: Autonomous Retraining Recommendation
    recommendation = recommender.recommend_action(drift_analysis, verified_results)
    
    # Print final decision
    recommender.print_recommendation(recommendation)
    
    return {
        'motion_regions': len(motion_regions),
        'total_keyframes_sampled': len(keyframes),
        'frames_selected': len(selected_frames),
        'frame_selection_method': frame_selection_method,
        'classifications': classifications,
        'verification': verified_results,
        'drift_analysis': drift_analysis,
        'recommendation': recommendation
    }


# ============================================================================
# USAGE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("AMBIENT AGENT CONFIGURATION")
    print("="*70)
    
    # Configure paths
    video_path = './video-assets/normal/1.mp4'  # Your test video
    classifier_model_path = './classifier-models/train-fox-background/weights/best.pt'
    save_keyframes_dir = './agent_keyframes'
    save_classified_dir = './agent_classified'  # NEW: Save classified images by class
    
    # Configure selection method
    # Options: 'by_region', 'balanced', 'motion_intensity', 'quality'
    frame_selection_method = 'by_region'  # Recommended: one best frame per motion region
    num_select = 5  # Only used for non-'by_region' methods
    
    print(f"Video: {video_path}")
    print(f"Model: {classifier_model_path}")
    print(f"Save keyframes: {save_keyframes_dir}")
    print(f"Save classified images: {save_classified_dir}")
    print(f"Frame selection: {frame_selection_method}")
    print(f"Num select: {num_select}")
    
    # Run agent
    if os.path.exists(video_path) and os.path.exists(classifier_model_path):
        results = ambient_monitoring_agent(
            video_path, 
            classifier_model_path,
            save_keyframes_dir=save_keyframes_dir,
            save_classified_dir=save_classified_dir,
            frame_selection_method=frame_selection_method,
            num_select=num_select
        )
        
        if results:
            # Print summary
            print("\n" + "="*70)
            print("AGENT EXECUTION SUMMARY")
            print("="*70)
            print(f"Motion regions detected: {results['motion_regions']}")
            print(f"Total keyframes sampled: {results['total_keyframes_sampled']}")
            print(f"Frames selected for classification: {results['frames_selected']}")
            print(f"Reduction: {results['total_keyframes_sampled']} → {results['frames_selected']} "
                  f"({100*(results['total_keyframes_sampled']-results['frames_selected'])/results['total_keyframes_sampled']:.0f}%)")
            print(f"Method: {results['frame_selection_method']}")
            
            # Print directory structure
            print(f"\n{'='*70}")
            print("OUTPUT DIRECTORY STRUCTURE")
            print(f"{'='*70}")
            print(f"\nKeyframes (before classification):")
            print(f"  {save_keyframes_dir}/")
            print(f"  ├── region_00_sample_00_frame_XXXX_ts_X.XXs.jpg")
            print(f"  ├── region_00_sample_01_frame_XXXX_ts_X.XXs.jpg")
            print(f"  └── ... (all sampled keyframes)")
            
            print(f"\nClassified Images (organized by predicted class):")
            print(f"  {save_classified_dir}/")
            if os.path.exists(save_classified_dir):
                for class_name in sorted(os.listdir(save_classified_dir)):
                    class_path = os.path.join(save_classified_dir, class_name)
                    if os.path.isdir(class_path):
                        num_files = len(os.listdir(class_path))
                        print(f"  ├── {class_name}/ ({num_files} images)")
            
            # Save analysis
            results_cleaned = {
                'summary': {
                    'motion_regions': results['motion_regions'],
                    'total_keyframes': results['total_keyframes_sampled'],
                    'selected_frames': results['frames_selected'],
                    'reduction_percentage': 100*(results['total_keyframes_sampled']-results['frames_selected'])/results['total_keyframes_sampled'],
                    'selection_method': results['frame_selection_method']
                },
                'drift_analysis': results['drift_analysis'],
                'recommendation': results['recommendation']
            }
            
            with open('agent_analysis.json', 'w') as f:
                json.dump(results_cleaned, f, indent=2)
            
            print("\n" + "="*70)
            print("FILES SAVED")
            print("="*70)
            print(f"✓ Analysis: agent_analysis.json")
            print(f"✓ Keyframes: {save_keyframes_dir}/")
            print(f"✓ Classified images: {save_classified_dir}/")
            
    else:
        print(f"Missing files:")
        if not os.path.exists(video_path):
            print(f"  - Video: {video_path}")
        if not os.path.exists(classifier_model_path):
            print(f"  - Model: {classifier_model_path}")