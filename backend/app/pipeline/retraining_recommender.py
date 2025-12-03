"""
Phase 6: Autonomous Retraining Recommendation
"""

from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class RetrainingRecommender:
    """Make autonomous retraining decisions based on classifier and VLM analysis"""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize recommender.

        Args:
            confidence_threshold: Threshold below which retraining is recommended
        """
        self.confidence_threshold = confidence_threshold
        self.history: List[Dict[str, Any]] = []

    def recommend_action(
        self,
        drift_analysis: Dict[str, Any],
        verified_results: List[Dict[str, Any]],
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Make autonomous recommendation based on classifier confidence and VLM validation.

        The recommendation is based on:
        1. Average classifier confidence
        2. VLM validation rate (valid vs invalid)
        3. Aggregate confidence = (avg_classifier_conf * vlm_validity_rate)

        If aggregate confidence < 0.7, recommend retraining.

        Args:
            drift_analysis: Results from drift detection
            verified_results: List of VLM verification results

        Returns:
            Recommendation dictionary
        """
        if on_progress:
            on_progress({
                "phase": "retraining_recommendation",
                "status": "started",
                "drift_score": drift_analysis.get('drift_score', 0),
                "drift_detected": drift_analysis.get('drift_detected', False)
            })

        # Calculate metrics from verified results
        total_verified = len(verified_results)
        valid_count = sum(1 for v in verified_results if v.get('class_valid', False))
        invalid_count = total_verified - valid_count

        # Calculate average classifier confidence
        avg_classifier_conf = sum(
            v.get('classifier_confidence', 0) for v in verified_results
        ) / total_verified if total_verified > 0 else 0

        # VLM validity rate
        vlm_validity_rate = valid_count / total_verified if total_verified > 0 else 0

        # Aggregate confidence = classifier confidence weighted by VLM validity
        aggregate_confidence = avg_classifier_conf * vlm_validity_rate

        # Get invalid classifications for the recommendation
        invalid_frames = [
            {
                'frame_idx': v['frame_idx'],
                'predicted_class': v['classifier_prediction'],
                'confidence': v['classifier_confidence'],
                'observation': v.get('vlm_observation', '')
            }
            for v in verified_results if not v.get('class_valid', False)
        ]

        # Determine recommendation based on aggregate confidence threshold
        if aggregate_confidence < self.confidence_threshold:
            if vlm_validity_rate < 0.5:
                # More than half are invalid
                recommendation = 'URGENT_RETRAIN'
                confidence = 0.95
                rationale = (
                    f"Critical: Only {vlm_validity_rate:.0%} of classifications validated by VLM. "
                    f"Aggregate confidence ({aggregate_confidence:.1%}) is well below threshold ({self.confidence_threshold:.0%}). "
                    "Model is producing unreliable predictions."
                )
                actions = [
                    "Stop using model for production immediately",
                    f"Review {invalid_count} misclassified frames",
                    "Collect new training samples for failed classes",
                    "Retrain model with expanded dataset",
                    "Consider if new object classes need to be added"
                ]
            else:
                recommendation = 'RETRAIN_RECOMMENDED'
                confidence = 0.8
                rationale = (
                    f"Aggregate confidence ({aggregate_confidence:.1%}) is below threshold ({self.confidence_threshold:.0%}). "
                    f"VLM validated {vlm_validity_rate:.0%} of predictions. "
                    "Model retraining recommended to improve accuracy."
                )
                actions = [
                    f"Review {invalid_count} misclassified frames for patterns",
                    "Collect additional training samples",
                    "Schedule model retraining",
                    "Monitor classification quality after retraining"
                ]
        else:
            if aggregate_confidence > 0.85:
                recommendation = 'CONTINUE_MONITORING'
                confidence = 0.9
                rationale = (
                    f"Model performing well. Aggregate confidence ({aggregate_confidence:.1%}) "
                    f"exceeds threshold. VLM validated {vlm_validity_rate:.0%} of predictions."
                )
                actions = [
                    "No immediate action required",
                    "Continue standard monitoring",
                    "Periodically validate with VLM"
                ]
            else:
                recommendation = 'MONITOR_CLOSELY'
                confidence = 0.75
                rationale = (
                    f"Aggregate confidence ({aggregate_confidence:.1%}) is above threshold but not optimal. "
                    f"VLM validated {vlm_validity_rate:.0%} of predictions. Watch for degradation."
                )
                actions = [
                    "Increase monitoring frequency",
                    "Prepare retraining pipeline",
                    f"Review {invalid_count} edge cases"
                ]

        decision = {
            'timestamp': datetime.now().isoformat(),
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale,
            'actions': actions,
            'metrics': {
                'aggregate_confidence': round(aggregate_confidence, 3),
                'avg_classifier_confidence': round(avg_classifier_conf, 3),
                'vlm_validity_rate': round(vlm_validity_rate, 3),
                'threshold': self.confidence_threshold,
                'total_verified': total_verified,
                'valid_count': valid_count,
                'invalid_count': invalid_count
            },
            'invalid_frames': invalid_frames,
            'drift_score': drift_analysis.get('drift_score', 0),
            'mismatch_rate': drift_analysis.get('mismatch_rate', 0),
            'verified_samples': total_verified,
            'drift_detected': drift_analysis.get('drift_detected', False)
        }

        self.history.append(decision)

        if on_progress:
            on_progress({
                "phase": "retraining_recommendation",
                "status": "completed",
                **decision
            })

        logger.info(
            f"Recommendation: {recommendation} "
            f"(aggregate confidence: {aggregate_confidence:.1%}, "
            f"threshold: {self.confidence_threshold:.0%})"
        )

        return decision

    def get_history(self) -> List[Dict[str, Any]]:
        """Get recommendation history"""
        return self.history.copy()
