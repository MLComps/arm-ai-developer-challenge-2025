"""
Phase 2.5: Intelligent Frame Selection
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class FrameSelector:
    """Intelligently select interesting keyframes before classification"""

    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate frame sharpness using Laplacian variance"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def calculate_entropy(frame: np.ndarray) -> float:
        """Calculate frame entropy (information content)"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)

    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate frame contrast"""
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    @staticmethod
    def select_diverse_frames(
        keyframes: List[Dict[str, Any]],
        num_select: int = 5,
        method: str = 'motion_intensity',
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select diverse keyframes using different strategies.

        Args:
            keyframes: List of keyframe dictionaries
            num_select: Number of frames to select
            method: Selection method ('motion_intensity', 'quality', 'entropy', 'contrast', 'balanced')
            on_progress: Callback for progress updates

        Returns:
            Selected keyframes
        """
        if len(keyframes) <= num_select:
            return keyframes

        if on_progress:
            on_progress({
                "phase": "frame_selection",
                "status": "started",
                "total_keyframes": len(keyframes),
                "selecting": num_select,
                "method": method
            })

        # Calculate metrics for all frames
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

            if on_progress and (i + 1) % 5 == 0:
                on_progress({
                    "phase": "frame_selection",
                    "status": "calculating_metrics",
                    "current": i + 1,
                    "total": len(keyframes),
                    "progress_percent": round(((i + 1) / len(keyframes)) * 50, 1)
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

        # Calculate final scores based on method
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
            metric['score'] = float(scores[i])

        # Sort by score and select top frames
        metrics.sort(key=lambda x: x['score'], reverse=True)
        selected = metrics[:num_select]

        # Re-sort by region and frame order for consistency
        selected.sort(key=lambda x: (x['keyframe']['region_id'], x['keyframe']['frame_idx']))

        result = [m['keyframe'] for m in selected]

        reduction_percent = round(100 * (len(keyframes) - len(result)) / len(keyframes), 1)

        if on_progress:
            on_progress({
                "phase": "frame_selection",
                "status": "completed",
                "selected_frames": len(result),
                "reduction_percent": reduction_percent,
                "method": method
            })

        logger.info(f"Selected {len(result)} frames ({reduction_percent}% reduction)")

        return result

    @staticmethod
    def select_by_region(
        keyframes: List[Dict[str, Any]],
        max_per_region: int = 1,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select best frame(s) from each motion region.

        Args:
            keyframes: List of keyframe dictionaries
            max_per_region: Maximum frames to select per region
            on_progress: Callback for progress updates

        Returns:
            Selected keyframes
        """
        if on_progress:
            on_progress({
                "phase": "frame_selection",
                "status": "started",
                "total_keyframes": len(keyframes),
                "max_per_region": max_per_region,
                "method": "by_region"
            })

        # Group by region
        regions = defaultdict(list)
        for kf in keyframes:
            regions[kf['region_id']].append(kf)

        selected = []
        for region_id in sorted(regions.keys()):
            region_frames = sorted(
                regions[region_id],
                key=lambda x: x['motion_intensity'],
                reverse=True
            )
            top = region_frames[:max_per_region]
            selected.extend(top)

            logger.debug(f"Region {region_id}: Selected {len(top)} frame(s)")

        reduction_percent = round(100 * (len(keyframes) - len(selected)) / len(keyframes), 1) if keyframes else 0

        if on_progress:
            on_progress({
                "phase": "frame_selection",
                "status": "completed",
                "selected_frames": len(selected),
                "total_regions": len(regions),
                "reduction_percent": reduction_percent,
                "method": "by_region"
            })

        logger.info(f"Selected {len(selected)} frames from {len(regions)} regions ({reduction_percent}% reduction)")

        return selected
