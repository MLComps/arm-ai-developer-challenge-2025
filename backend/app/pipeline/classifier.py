"""
Phase 3: Frame Classification using YOLO
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class FrameClassifier:
    """Classify sampled keyframes using YOLO model"""

    def __init__(self, model_path: str):
        """
        Initialize classifier with YOLO model.

        Args:
            model_path: Path to YOLO model weights
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)

        # Handle both dict and list formats for class names
        if hasattr(self.model, 'names'):
            if isinstance(self.model.names, dict):
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self.model.names
        else:
            self.class_names = ['background', 'fox', 'coati', 'ocelot', 'deer']

        logger.info(f"Loaded model with classes: {self.class_names}")

    def classify_keyframes(
        self,
        keyframes: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify each keyframe.

        Args:
            keyframes: List of keyframe dictionaries with 'frame' key
            save_dir: Optional directory to save classified images by class
            on_progress: Callback for progress updates

        Returns:
            List of classification result dictionaries
        """
        if on_progress:
            on_progress({
                "phase": "classification",
                "status": "started",
                "total_frames": len(keyframes),
                "classes": self.class_names
            })

        # Create save directories if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for class_name in self.class_names:
                class_dir = os.path.join(save_dir, str(class_name).replace(' ', '_'))
                os.makedirs(class_dir, exist_ok=True)
            logger.info(f"Saving classified images to: {save_dir}")

        classifications = []

        for idx, keyframe_data in enumerate(keyframes):
            frame = keyframe_data['frame']

            # Run prediction
            results = self.model.predict(frame, verbose=False)
            pred = results[0]

            probs = pred.probs.data.cpu().numpy()
            predicted_class_id = int(np.argmax(probs))
            confidence = float(probs[predicted_class_id])

            if predicted_class_id < len(self.class_names):
                predicted_class = str(self.class_names[predicted_class_id])
            else:
                predicted_class = f"class_{predicted_class_id}"

            # Save image if directory specified
            saved_path = None
            if save_dir:
                class_dir = os.path.join(save_dir, predicted_class.replace(' ', '_'))
                filename = (
                    f"region_{keyframe_data['region_id']:02d}_"
                    f"frame_{keyframe_data['frame_idx']:04d}_"
                    f"ts_{keyframe_data['timestamp']:.2f}s_"
                    f"class_{predicted_class.replace(' ', '_')}_"
                    f"conf_{confidence:.3f}.jpg"
                )
                saved_path = os.path.join(class_dir, filename)
                cv2.imwrite(saved_path, frame)

            # Build all probabilities dict
            all_probs = {}
            for i in range(len(probs)):
                if i < len(self.class_names):
                    all_probs[str(self.class_names[i])] = float(probs[i])
                else:
                    all_probs[f'class_{i}'] = float(probs[i])

            classification = {
                'frame_idx': keyframe_data['frame_idx'],
                'timestamp': keyframe_data['timestamp'],
                'region_id': keyframe_data['region_id'],
                'predicted_class': predicted_class,
                'class_id': predicted_class_id,
                'confidence': confidence,
                'all_probs': all_probs,
                'frame': frame,
                'saved_path': saved_path
            }

            classifications.append(classification)

            if on_progress:
                on_progress({
                    "phase": "classification",
                    "status": "processing",
                    "current_frame": idx + 1,
                    "total_frames": len(keyframes),
                    "progress_percent": round(((idx + 1) / len(keyframes)) * 100, 1),
                    "last_prediction": {
                        "frame_idx": keyframe_data['frame_idx'],
                        "class": predicted_class,
                        "confidence": round(confidence, 3),
                        "saved_path": saved_path
                    }
                })

            logger.debug(
                f"Frame {keyframe_data['frame_idx']} @ {keyframe_data['timestamp']:.2f}s "
                f"-> {predicted_class} (conf: {confidence:.3f})"
            )

        if on_progress:
            # Summarize predictions
            class_counts = {}
            for clf in classifications:
                cls = clf['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            on_progress({
                "phase": "classification",
                "status": "completed",
                "total_classified": len(classifications),
                "class_distribution": class_counts,
                "save_directory": save_dir
            })

        logger.info(f"Classified {len(classifications)} frames")

        return classifications
