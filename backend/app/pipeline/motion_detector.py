"""
Phase 1: Motion Detection using MOG2 Background Subtraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class MotionDetector:
    """MOG2 Background Subtraction - optimized for camera traps"""

    def __init__(self, history: int = 500, var_threshold: int = 16):
        self.history = history
        self.var_threshold = var_threshold

    def _create_subtractor(self):
        """Create a fresh background subtractor instance"""
        return cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            history=self.history,
            varThreshold=self.var_threshold
        )

    def detect_motion_regions(
        self,
        video_path: str,
        motion_threshold: float = 0.02,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Detect motion regions in video.

        Args:
            video_path: Path to video file
            motion_threshold: Minimum motion intensity threshold (0-1)
            on_progress: Callback for progress updates

        Returns:
            List of tuples (start_frame, end_frame, max_intensity)
        """
        bg_subtractor = self._create_subtractor()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            if on_progress:
                on_progress({
                    "phase": "motion_detection",
                    "status": "error",
                    "message": f"Cannot open video: {video_path}"
                })
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if on_progress:
            on_progress({
                "phase": "motion_detection",
                "status": "started",
                "total_frames": total_frames,
                "fps": fps,
                "threshold": motion_threshold
            })

        motion_frames = []
        frame_idx = 0

        logger.info(f"Processing {total_frames} frames @ {fps} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)

            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Calculate motion intensity as percentage of frame
            motion_intensity = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])

            if motion_intensity > motion_threshold:
                motion_frames.append((frame_idx, motion_intensity))

            frame_idx += 1

            # Report progress every 50 frames
            if frame_idx % 50 == 0 and on_progress:
                on_progress({
                    "phase": "motion_detection",
                    "status": "processing",
                    "current_frame": frame_idx,
                    "total_frames": total_frames,
                    "progress_percent": round((frame_idx / total_frames) * 100, 1),
                    "motion_frames_found": len(motion_frames)
                })

        cap.release()

        # Group consecutive motion frames into regions
        motion_regions = self._group_motion_frames(motion_frames, gap_threshold=5)

        if on_progress:
            on_progress({
                "phase": "motion_detection",
                "status": "completed",
                "total_frames_processed": frame_idx,
                "motion_frames": len(motion_frames),
                "motion_regions": len(motion_regions),
                "motion_percentage": round((len(motion_frames) / total_frames) * 100, 1) if total_frames > 0 else 0
            })

        logger.info(f"Found {len(motion_frames)} motion frames in {len(motion_regions)} regions")

        return motion_regions

    @staticmethod
    def _group_motion_frames(
        motion_frames: List[Tuple[int, float]],
        gap_threshold: int = 5
    ) -> List[Tuple[int, int, float]]:
        """
        Group consecutive motion frames into regions.

        Args:
            motion_frames: List of (frame_idx, intensity) tuples
            gap_threshold: Max frames between motion to consider same region

        Returns:
            List of (start_frame, end_frame, max_intensity) tuples
        """
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
                # Gap detected - save current region and start new one
                regions.append((region_start, prev_frame, region_max_intensity))
                region_start = current_frame
                region_max_intensity = current_intensity

        # Don't forget the last region
        regions.append((region_start, motion_frames[-1][0], region_max_intensity))

        return regions
