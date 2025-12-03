"""
Phase 5: Data Drift Detection
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift from verification mismatches"""

    def __init__(self, mismatch_threshold: float = 0.3):
        """
        Initialize drift detector.

        Args:
            mismatch_threshold: Threshold for drift detection (0-1)
        """
        self.mismatch_threshold = mismatch_threshold

    def detect_drift(
        self,
        verified_results: List[Dict[str, Any]],
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyze verification results to detect data drift.

        Args:
            verified_results: List of verification results
            on_progress: Callback for progress updates

        Returns:
            Drift analysis dictionary
        """
        if on_progress:
            on_progress({
                "phase": "drift_detection",
                "status": "started",
                "total_results": len(verified_results),
                "mismatch_threshold": self.mismatch_threshold
            })

        if not verified_results:
            result = {
                'drift_detected': False,
                'drift_score': 0.0,
                'mismatch_rate': 0.0,
                'avg_confidence': 0.0,
                'class_dominance': 0.0,
                'predictions': {}
            }
            if on_progress:
                on_progress({
                    "phase": "drift_detection",
                    "status": "completed",
                    **result
                })
            return result

        # Calculate invalid rate (using class_valid field from VLM verifier)
        # Support both 'class_valid' (new) and 'match' (old) field names
        invalid_count = sum(
            1 for r in verified_results
            if not r.get('class_valid', r.get('match', True))
        )
        mismatch_rate = invalid_count / len(verified_results)

        # Calculate average classifier confidence
        confidences = [
            r.get('classifier_confidence', r.get('confidence', 0.5))
            for r in verified_results
        ]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.5

        # Calculate prediction distribution
        predictions = defaultdict(int)
        for r in verified_results:
            pred_class = r.get('classifier_prediction', r.get('predicted_class', 'unknown'))
            predictions[pred_class] += 1

        total_preds = len(verified_results)
        max_class_count = max(predictions.values()) if predictions else 0
        class_dominance = max_class_count / total_preds if total_preds > 0 else 0

        # Calculate overall drift score
        # Higher score = more likely drift
        # Factors: mismatch rate, low confidence, class imbalance
        low_conf_penalty = max(0, 0.7 - avg_confidence)  # Penalty if avg conf < 0.7
        drift_score = (
            mismatch_rate * 0.5 +
            low_conf_penalty * 0.3 +
            ((class_dominance - 0.7) * 0.2 if class_dominance > 0.7 else 0)
        )

        # Determine if drift is detected
        drift_detected = (
            mismatch_rate > self.mismatch_threshold or
            class_dominance > 0.9 or
            avg_confidence < 0.5
        )

        result = {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'mismatch_rate': float(mismatch_rate),
            'avg_confidence': float(avg_confidence),
            'class_dominance': float(class_dominance),
            'predictions': dict(predictions),
            'total_samples': total_preds,
            'invalid_count': invalid_count
        }

        if on_progress:
            on_progress({
                "phase": "drift_detection",
                "status": "completed",
                **result
            })

        status = "DRIFT DETECTED" if drift_detected else "No drift"
        logger.info(
            f"{status} - Score: {drift_score:.3f}, "
            f"Invalid rate: {mismatch_rate:.1%}, "
            f"Avg confidence: {avg_confidence:.1%}, "
            f"Class dominance: {class_dominance:.1%}"
        )

        return result
