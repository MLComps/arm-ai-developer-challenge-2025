"""
Video Processor Service - Orchestrates the complete pipeline
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import uuid
import logging

from ..pipeline import (
    MotionDetector,
    KeyframeSampler,
    FrameSelector,
    FrameClassifier,
    VLMVerifier,
    DataDriftDetector,
    RetrainingRecommender
)
from ..models.schemas import JobStatus, FrameSelectionMethod

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Orchestrates the complete video processing pipeline"""

    def __init__(
        self,
        classifier_model_path: str,
        output_base_dir: str = "./output",
        vlm_endpoint: Optional[str] = None,
        vlm_model: str = "qwen3-vl:2b",
        vlm_num_samples: int = 5
    ):
        """
        Initialize video processor.

        Args:
            classifier_model_path: Path to YOLO classifier model
            output_base_dir: Base directory for output files
            vlm_endpoint: VLM service endpoint
            vlm_model: VLM model name
            vlm_num_samples: Number of frames to sample for VLM verification
        """
        self.classifier_model_path = classifier_model_path
        self.output_base_dir = output_base_dir
        self.vlm_endpoint = vlm_endpoint

        # Initialize pipeline components
        self.motion_detector = MotionDetector()
        self.sampler = KeyframeSampler()
        self.frame_selector = FrameSelector()
        self.classifier = None  # Lazy load
        self.vlm_verifier = VLMVerifier(
            vlm_endpoint=vlm_endpoint,
            model_name=vlm_model,
            num_samples=vlm_num_samples
        )
        self.drift_detector = DataDriftDetector(mismatch_threshold=0.3)
        self.recommender = RetrainingRecommender()

        # Job tracking
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def _get_classifier(self, model_path: Optional[str] = None) -> FrameClassifier:
        """Lazy load classifier model, or reload if different model requested"""
        target_path = model_path or self.classifier_model_path

        # Reload classifier if a different model is requested
        if self.classifier is None or (model_path and model_path != self.classifier_model_path):
            logger.info(f"Loading classifier model: {target_path}")
            self.classifier = FrameClassifier(target_path)
            self.classifier_model_path = target_path

        return self.classifier

    def _create_job(self, video_path: str) -> str:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())[:8]
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "video_path": video_path,
            "started_at": datetime.now(),
            "completed_at": None,
            "current_phase": None,
            "progress_percent": 0,
            "results": None,
            "error": None
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job info by ID"""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list:
        """Get all jobs"""
        return list(self.jobs.values())

    async def process_video(
        self,
        video_path: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        frame_selection_method: FrameSelectionMethod = FrameSelectionMethod.BY_REGION,
        num_select: int = 5,
        samples_per_region: int = 3,
        motion_threshold: float = 0.02,
        save_keyframes: bool = True,
        save_classified: bool = True,
        classifier_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a video through the complete pipeline.

        Args:
            video_path: Path to video file
            on_event: Async callback for progress events
            frame_selection_method: Method for frame selection
            num_select: Number of frames to select
            samples_per_region: Keyframes per motion region
            motion_threshold: Motion detection threshold
            save_keyframes: Whether to save keyframe images
            save_classified: Whether to save classified images

        Returns:
            Processing results dictionary
        """
        job_id = self._create_job(video_path)
        job = self.jobs[job_id]

        # Setup output directories
        video_name = Path(video_path).stem
        job_output_dir = os.path.join(self.output_base_dir, f"job_{job_id}_{video_name}")
        keyframes_dir = os.path.join(job_output_dir, "keyframes") if save_keyframes else None
        classified_dir = os.path.join(job_output_dir, "classified") if save_classified else None

        if keyframes_dir:
            os.makedirs(keyframes_dir, exist_ok=True)
        if classified_dir:
            os.makedirs(classified_dir, exist_ok=True)

        # Get current event loop for thread-safe callbacks
        loop = asyncio.get_running_loop()

        def sync_progress_callback(data: Dict[str, Any]):
            """Synchronous wrapper for progress updates (called from threads)"""
            job["current_phase"] = data.get("phase")
            if "progress_percent" in data:
                job["progress_percent"] = data["progress_percent"]

            if on_event:
                # Schedule async event emission on the main event loop
                asyncio.run_coroutine_threadsafe(
                    self._emit_event(on_event, job_id, data),
                    loop
                )

        async def emit(data: Dict[str, Any]):
            """Emit event asynchronously"""
            if on_event:
                await self._emit_event(on_event, job_id, data)

        try:
            job["status"] = JobStatus.PROCESSING

            await emit({
                "event_type": "job_started",
                "job_id": job_id,
                "video_path": video_path
            })

            # Phase 1: Motion Detection
            logger.info(f"[{job_id}] Phase 1: Motion Detection")
            motion_regions = await asyncio.to_thread(
                self.motion_detector.detect_motion_regions,
                video_path,
                motion_threshold,
                sync_progress_callback
            )

            if not motion_regions:
                await emit({
                    "event_type": "warning",
                    "phase": "motion_detection",
                    "message": "No motion detected in video"
                })
                job["status"] = JobStatus.COMPLETED
                job["completed_at"] = datetime.now()
                job["results"] = {
                    "motion_regions": 0,
                    "message": "No motion detected"
                }
                return job

            # Phase 2: Keyframe Sampling
            logger.info(f"[{job_id}] Phase 2: Keyframe Sampling")
            keyframes = await asyncio.to_thread(
                self.sampler.sample_keyframes,
                video_path,
                motion_regions,
                samples_per_region,
                keyframes_dir,
                sync_progress_callback
            )

            # Phase 2.5: Frame Selection
            logger.info(f"[{job_id}] Phase 2.5: Frame Selection")
            if frame_selection_method == FrameSelectionMethod.BY_REGION:
                selected_frames = await asyncio.to_thread(
                    self.frame_selector.select_by_region,
                    keyframes,
                    1,  # max_per_region
                    sync_progress_callback
                )
            else:
                selected_frames = await asyncio.to_thread(
                    self.frame_selector.select_diverse_frames,
                    keyframes,
                    num_select,
                    frame_selection_method.value,
                    sync_progress_callback
                )

            # Phase 3: Classification
            logger.info(f"[{job_id}] Phase 3: Classification")
            classifier = self._get_classifier(classifier_model_path)
            classifications = await asyncio.to_thread(
                classifier.classify_keyframes,
                selected_frames,
                classified_dir,
                sync_progress_callback
            )

            # Phase 4: VLM Verification
            logger.info(f"[{job_id}] Phase 4: VLM Verification")
            verified_results = await asyncio.to_thread(
                self.vlm_verifier.verify_predictions,
                classifications,
                sync_progress_callback
            )

            # Phase 5: Drift Detection
            logger.info(f"[{job_id}] Phase 5: Drift Detection")
            drift_analysis = await asyncio.to_thread(
                self.drift_detector.detect_drift,
                verified_results,
                sync_progress_callback
            )

            # Phase 6: Retraining Recommendation
            logger.info(f"[{job_id}] Phase 6: Retraining Recommendation")
            recommendation = await asyncio.to_thread(
                self.recommender.recommend_action,
                drift_analysis,
                verified_results,
                sync_progress_callback
            )

            # Build results
            total_keyframes = len(keyframes)
            frames_selected = len(selected_frames)
            reduction_pct = round(
                100 * (total_keyframes - frames_selected) / total_keyframes, 1
            ) if total_keyframes > 0 else 0

            results = {
                "job_id": job_id,
                "video_path": video_path,
                "output_directory": job_output_dir,
                "summary": {
                    "motion_regions": len(motion_regions),
                    "total_keyframes_sampled": total_keyframes,
                    "frames_selected": frames_selected,
                    "frame_selection_method": frame_selection_method.value,
                    "reduction_percentage": reduction_pct
                },
                "motion_regions": [
                    {
                        "region_id": i,
                        "start_frame": r[0],
                        "end_frame": r[1],
                        "max_intensity": r[2]
                    }
                    for i, r in enumerate(motion_regions)
                ],
                "classifications": [
                    {
                        "frame_idx": c["frame_idx"],
                        "timestamp": c["timestamp"],
                        "region_id": c["region_id"],
                        "predicted_class": c["predicted_class"],
                        "class_id": c["class_id"],
                        "confidence": c["confidence"],
                        "all_probs": c["all_probs"],
                        "saved_path": c["saved_path"]
                    }
                    for c in classifications
                ],
                "verifications": [
                    {
                        "frame_idx": v["frame_idx"],
                        "timestamp": v["timestamp"],
                        "classifier_prediction": v["classifier_prediction"],
                        "classifier_confidence": v["classifier_confidence"],
                        "class_valid": v["class_valid"],
                        "vlm_confidence": v.get("vlm_confidence", 0.5),
                        "verification_status": v["verification_status"],
                        "vlm_observation": v.get("vlm_observation", ""),
                        "saved_path": v.get("saved_path")
                    }
                    for v in verified_results
                ],
                "drift_analysis": drift_analysis,
                "recommendation": recommendation
            }

            job["status"] = JobStatus.COMPLETED
            job["completed_at"] = datetime.now()
            job["progress_percent"] = 100
            job["results"] = results

            await emit({
                "event_type": "job_completed",
                "job_id": job_id,
                "summary": results["summary"],
                "recommendation": recommendation["recommendation"]
            })

            logger.info(f"[{job_id}] Processing completed successfully")

            return job

        except Exception as e:
            logger.exception(f"[{job_id}] Processing failed: {e}")
            job["status"] = JobStatus.FAILED
            job["completed_at"] = datetime.now()
            job["error"] = str(e)

            await emit({
                "event_type": "job_failed",
                "job_id": job_id,
                "error": str(e)
            })

            return job

    async def _emit_event(
        self,
        callback: Callable[[Dict[str, Any]], None],
        job_id: str,
        data: Dict[str, Any]
    ):
        """Emit event through callback"""
        try:
            event = {
                "job_id": job_id,
                **data
            }
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")


# Singleton processor instance (will be initialized with config)
_processor: Optional[VideoProcessor] = None


def get_processor() -> VideoProcessor:
    """Get the global processor instance"""
    global _processor
    if _processor is None:
        raise RuntimeError("VideoProcessor not initialized. Call init_processor() first.")
    return _processor


def init_processor(
    classifier_model_path: str,
    output_base_dir: str = "./output",
    vlm_endpoint: Optional[str] = None,
    vlm_model: str = "qwen3-vl:2b",
    vlm_num_samples: int = 5
) -> VideoProcessor:
    """Initialize the global processor instance"""
    global _processor
    _processor = VideoProcessor(
        classifier_model_path=classifier_model_path,
        output_base_dir=output_base_dir,
        vlm_endpoint=vlm_endpoint,
        vlm_model=vlm_model,
        vlm_num_samples=vlm_num_samples
    )
    return _processor
