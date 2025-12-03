"""
Phase 4: VLM Verification of Classifier Predictions using Ollama
"""

import base64
import cv2
import json
import random
import requests
from typing import List, Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class VLMVerifier:
    """Verify classifier predictions with Vision Language Model (Ollama)"""

    def __init__(
        self,
        vlm_endpoint: Optional[str] = "http://localhost:11434/api/generate",
        model_name: str = "qwen3-vl:2b",
        num_samples: int = 5
    ):
        """
        Initialize VLM verifier.

        Args:
            vlm_endpoint: Ollama API endpoint
            model_name: VLM model to use
            num_samples: Number of frames to sample for VLM verification
        """
        self.vlm_endpoint = vlm_endpoint
        self.model_name = model_name
        self.num_samples = num_samples

        if vlm_endpoint:
            logger.info(f"VLM endpoint configured: {vlm_endpoint} with model {model_name}")
        else:
            logger.warning("No VLM endpoint configured - using mock verification")

    def _encode_image_to_base64(self, frame) -> str:
        """Convert CV2 frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def _sample_frames_for_verification(
        self,
        classifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sample frames for VLM verification.
        Prioritizes lowest confidence predictions with some randomness.

        Args:
            classifications: All classified frames

        Returns:
            Selected frames for VLM verification
        """
        if len(classifications) <= self.num_samples:
            return classifications

        # Sort by confidence (ascending - lowest first)
        sorted_by_conf = sorted(classifications, key=lambda x: x['confidence'])

        # Take the 3 lowest confidence frames
        lowest_conf = sorted_by_conf[:3]

        # Randomly sample 2 more from the remaining
        remaining = sorted_by_conf[3:]
        if len(remaining) >= 2:
            random_samples = random.sample(remaining, 2)
        else:
            random_samples = remaining

        selected = lowest_conf + random_samples
        logger.info(f"Sampled {len(selected)} frames for VLM verification "
                   f"(3 lowest conf + {len(random_samples)} random)")

        return selected

    def _call_vlm(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Ollama VLM endpoint for verification.

        Args:
            classification: Classification result with frame

        Returns:
            VLM assessment dictionary
        """
        predicted_class = classification['predicted_class']
        frame = classification.get('frame')

        if frame is None:
            logger.warning("No frame data available for VLM verification")
            return self._create_error_response("No frame data available")

        # Encode image to base64
        image_b64 = self._encode_image_to_base64(frame)

        # Create prompt for VLM - context-aware prompt
        if predicted_class.lower() == 'background':
            prompt = f"""Look at this wildlife camera image carefully.

The classifier labeled this as: "background" (meaning NO ANIMAL is present)

Question: Is this correct? Is the image empty of animals, showing only background/environment?

Answer with:
1. YES if there is NO animal visible (classification is correct)
2. NO if you can see an animal (classification is wrong)

Format your response as:
VALID: YES or NO
OBSERVATION: briefly describe what you see (e.g., "empty forest scene" or "I can see a fox")"""
        else:
            prompt = f"""Look at this wildlife camera image carefully.

The classifier labeled this as: "{predicted_class}" (an animal)

Question: Is this correct? Can you see a {predicted_class} in the image?

Answer with:
1. YES if you can see a {predicted_class} in the image
2. NO if there is no {predicted_class} (either empty/background or different animal)

Format your response as:
VALID: YES or NO
OBSERVATION: briefly describe what you see"""

        try:
            response = requests.post(
                self.vlm_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            vlm_response = result.get('response', '')

            logger.info(f"VLM raw response for {predicted_class}: {vlm_response[:150] if vlm_response else '(empty)'}...")

            if not vlm_response:
                logger.warning("Empty VLM response received")
                return self._create_error_response("Empty response from VLM")

            # Parse the response
            return self._parse_vlm_response(vlm_response, predicted_class)

        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request failed: {e}")
            return self._create_error_response(str(e))

    def _parse_vlm_response(
        self,
        response: str,
        predicted_class: str
    ) -> Dict[str, Any]:
        """Parse VLM response looking for VALID: YES/NO pattern"""
        response_lower = response.lower()
        logger.debug(f"Parsing VLM response: {response[:200]}")

        # Try to find VALID: YES/NO pattern
        class_valid = None
        observation = response

        # Look for explicit VALID: YES/NO pattern
        if 'valid:' in response_lower:
            parts = response_lower.split('valid:')
            if len(parts) > 1:
                answer_part = parts[1].strip()[:20]
                if answer_part.startswith('yes') or 'yes' in answer_part[:10]:
                    class_valid = True
                elif answer_part.startswith('no') or 'no' in answer_part[:10]:
                    class_valid = False

        # Try to find OBSERVATION: pattern
        if 'observation:' in response_lower:
            parts = response.split('OBSERVATION:')
            if len(parts) < 2:
                parts = response.split('observation:')
            if len(parts) > 1:
                observation = parts[1].strip()

        # If we couldn't find the pattern, fall back to keyword detection
        if class_valid is None:
            class_valid = self._detect_validity_from_keywords(response_lower, predicted_class)

        # Clean up observation
        observation = observation.strip()[:200]
        if not observation:
            observation = "No detailed observation provided"

        return {
            'class_valid': class_valid,
            'detected_object': predicted_class if class_valid else 'unknown',
            'confidence': 0.85 if class_valid else 0.4,
            'observation': observation,
            'raw_response': response
        }

    def _detect_validity_from_keywords(
        self,
        response_lower: str,
        predicted_class: str
    ) -> bool:
        """Detect validity from keywords in response"""
        predicted_lower = predicted_class.lower()
        is_background = predicted_lower == 'background'

        # For background classification
        if is_background:
            # Valid if no animal is seen
            background_valid = [
                'no animal', 'empty', 'just background', 'only background',
                'no wildlife', 'environment only', 'forest scene', 'nature scene',
                'vegetation', 'trees', 'leaves', 'nothing visible',
                'valid: yes', 'yes'
            ]
            # Invalid if animal is seen
            background_invalid = [
                'see a', 'see an', 'there is a', 'there is an',
                'i can see', 'animal', 'fox', 'deer', 'coati', 'ocelot',
                'creature', 'wildlife visible', 'valid: no', 'no'
            ]

            for indicator in background_invalid:
                if indicator in response_lower:
                    # Make sure "no animal" doesn't trigger false negative
                    if indicator == 'no' and 'no animal' in response_lower:
                        continue
                    return False

            for indicator in background_valid:
                if indicator in response_lower:
                    return True

            return True  # Default to valid for background if uncertain

        # Strong positive indicators for animal classes
        strong_positive = [
            f'is a {predicted_lower}',
            f'shows a {predicted_lower}',
            f'contains a {predicted_lower}',
            f'see a {predicted_lower}',
            f'is {predicted_lower}',
            f'the {predicted_lower}',
            'classification is correct',
            'correct classification',
            'yes, this is',
            'yes, the image',
            'correctly classified',
            'valid: yes'
        ]

        # Strong negative indicators
        strong_negative = [
            'not a ' + predicted_lower,
            'no ' + predicted_lower,
            f'is not {predicted_lower}',
            f"isn't a {predicted_lower}",
            f"doesn't show a {predicted_lower}",
            'classification is incorrect',
            'incorrect classification',
            'misclassified',
            'wrongly classified',
            'no, this is',
            'no, the image',
            'valid: no',
            'cannot see',
            'do not see',
            "don't see",
            'no animal',
            'empty',
            'just background',
            'only background'
        ]

        # Check strong indicators first
        for indicator in strong_positive:
            if indicator in response_lower:
                return True

        for indicator in strong_negative:
            if indicator in response_lower:
                return False

        # Weaker keyword detection
        positive_words = ['yes', 'correct', 'right', 'accurate', 'confirmed', 'indeed']
        negative_words = ['no', 'incorrect', 'wrong', 'inaccurate', 'false', 'cannot', 'unable']

        pos_count = sum(1 for word in positive_words if word in response_lower)
        neg_count = sum(1 for word in negative_words if word in response_lower)

        if neg_count > pos_count:
            return False
        elif pos_count > neg_count:
            return True

        # Default to True if the predicted class is mentioned positively
        if predicted_lower in response_lower:
            # Check if it's mentioned in a negative context
            for neg in ['not', 'no', "n't", 'cannot', 'unable']:
                # Check if negative word is near the class mention
                idx = response_lower.find(predicted_lower)
                if idx > 0:
                    context = response_lower[max(0, idx-15):idx]
                    if neg in context:
                        return False
            return True

        # Default to False if uncertain
        return False

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create error response when VLM call fails"""
        return {
            'class_valid': False,
            'detected_object': 'error',
            'confidence': 0.0,
            'observation': f'VLM verification failed: {error}',
            'raw_response': None
        }

    def verify_predictions(
        self,
        classifications: List[Dict[str, Any]],
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Use VLM to validate classifier predictions.

        Args:
            classifications: List of classification results
            on_progress: Callback for progress updates

        Returns:
            List of verification result dictionaries
        """
        # Sample frames for verification
        sampled = self._sample_frames_for_verification(classifications)

        if on_progress:
            on_progress({
                "phase": "vlm_verification",
                "status": "started",
                "total_predictions": len(sampled),
                "using_mock": self.vlm_endpoint is None,
                "sampled_from": len(classifications)
            })

        verified_results = []
        valid_count = 0
        invalid_count = 0

        for idx, clf in enumerate(sampled):
            # Call VLM for verification
            if self.vlm_endpoint:
                vlm_assessment = self._call_vlm(clf)
            else:
                vlm_assessment = self._mock_vlm_assessment(clf)

            class_valid = vlm_assessment.get('class_valid', False)
            observation = vlm_assessment.get('observation', '')

            if class_valid:
                valid_count += 1
            else:
                invalid_count += 1

            result = {
                'frame_idx': clf['frame_idx'],
                'timestamp': clf['timestamp'],
                'classifier_prediction': clf['predicted_class'],
                'classifier_confidence': clf['confidence'],
                'class_valid': class_valid,
                'vlm_observation': observation,
                'vlm_confidence': vlm_assessment.get('confidence', 0.5),
                'saved_path': clf.get('saved_path'),
                'all_probs': clf.get('all_probs', {}),
                'verification_status': 'valid' if class_valid else 'invalid'
            }

            verified_results.append(result)

            if on_progress:
                on_progress({
                    "phase": "vlm_verification",
                    "status": "processing",
                    "current": idx + 1,
                    "total": len(sampled),
                    "progress_percent": round(((idx + 1) / len(sampled)) * 100, 1),
                    "valid_count": valid_count,
                    "invalid_count": invalid_count,
                    "last_verification": {
                        "frame_idx": clf['frame_idx'],
                        "classifier": clf['predicted_class'],
                        "classifier_confidence": round(clf['confidence'], 3),
                        "class_valid": class_valid,
                        "observation": observation[:100],
                        "saved_path": clf.get('saved_path')
                    }
                })

            status = "VALID" if class_valid else "INVALID"
            logger.info(
                f"Frame {clf['frame_idx']}: {clf['predicted_class']} "
                f"(conf: {clf['confidence']:.3f}) -> VLM: {status}"
            )

        # Calculate aggregate metrics
        total_verified = len(verified_results)
        validity_rate = (valid_count / total_verified * 100) if total_verified > 0 else 0

        # Calculate average classifier confidence for verified frames
        avg_classifier_conf = sum(
            r['classifier_confidence'] for r in verified_results
        ) / total_verified if total_verified > 0 else 0

        if on_progress:
            on_progress({
                "phase": "vlm_verification",
                "status": "completed",
                "total_verified": total_verified,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "validity_rate": round(validity_rate, 1),
                "avg_classifier_confidence": round(avg_classifier_conf, 3)
            })

        logger.info(f"VLM verification complete: {valid_count} valid, {invalid_count} invalid")

        return verified_results

    @staticmethod
    def _mock_vlm_assessment(classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock VLM response based on classifier confidence.
        Used when no VLM endpoint is configured.
        """
        classifier_conf = classification['confidence']
        predicted_class = classification['predicted_class']

        if classifier_conf > 0.9:
            return {
                'class_valid': True,
                'detected_object': predicted_class,
                'confidence': 0.95,
                'observation': f"Image clearly shows {predicted_class}"
            }
        elif classifier_conf > 0.7:
            return {
                'class_valid': True,
                'detected_object': predicted_class,
                'confidence': 0.8,
                'observation': f"Image likely contains {predicted_class}"
            }
        elif classifier_conf > 0.5:
            # Random chance of being invalid
            is_valid = random.random() > 0.4
            return {
                'class_valid': is_valid,
                'detected_object': predicted_class if is_valid else 'uncertain',
                'confidence': 0.5,
                'observation': "Classification uncertain - borderline confidence"
            }
        else:
            return {
                'class_valid': False,
                'detected_object': 'unknown',
                'confidence': 0.3,
                'observation': "Low confidence classification likely incorrect"
            }
