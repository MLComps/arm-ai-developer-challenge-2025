"""
Phase 2: Keyframe Sampling from Motion Regions
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KeyframeSampler:
    """Extract and save representative keyframes from motion regions"""

    @staticmethod
    def sample_keyframes(
        video_path: str,
        motion_regions: List[Tuple[int, int, float]],
        samples_per_region: int = 3,
        save_dir: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample keyframes with highest motion intensity from each region.

        Args:
            video_path: Path to video file
            motion_regions: List of (start, end, intensity) tuples
            samples_per_region: Number of frames to sample per region
            save_dir: Optional directory to save keyframe images
            on_progress: Callback for progress updates

        Returns:
            List of keyframe dictionaries with frame data and metadata
        """
        if on_progress:
            on_progress({
                "phase": "keyframe_sampling",
                "status": "started",
                "total_regions": len(motion_regions),
                "samples_per_region": samples_per_region
            })

        keyframes = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving keyframes to: {save_dir}")

        for region_idx, (start_frame, end_frame, region_intensity) in enumerate(motion_regions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            region_frames = []
            prev_gray = None

            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate motion intensity as frame difference
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

            # Sort by motion intensity and take top samples
            region_frames.sort(key=lambda x: x['motion_intensity'], reverse=True)
            top_frames = region_frames[:samples_per_region]

            for sample_idx, frame_data in enumerate(top_frames):
                timestamp = frame_data['frame_idx'] / fps if fps > 0 else 0

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

            if on_progress:
                on_progress({
                    "phase": "keyframe_sampling",
                    "status": "processing",
                    "current_region": region_idx + 1,
                    "total_regions": len(motion_regions),
                    "progress_percent": round(((region_idx + 1) / len(motion_regions)) * 100, 1),
                    "keyframes_sampled": len(keyframes)
                })

            logger.debug(f"Region {region_idx + 1}: Sampled {len(top_frames)} keyframes")

        cap.release()

        if on_progress:
            on_progress({
                "phase": "keyframe_sampling",
                "status": "completed",
                "total_keyframes": len(keyframes),
                "save_directory": save_dir
            })

        logger.info(f"Total keyframes sampled: {len(keyframes)}")

        return keyframes
