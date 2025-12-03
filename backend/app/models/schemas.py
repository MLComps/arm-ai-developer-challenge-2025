"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FrameSelectionMethod(str, Enum):
    BY_REGION = "by_region"
    BALANCED = "balanced"
    MOTION_INTENSITY = "motion_intensity"
    QUALITY = "quality"
    ENTROPY = "entropy"
    CONTRAST = "contrast"


# Request Models
class VideoProcessRequest(BaseModel):
    """Request to process a single video"""
    video_path: str = Field(..., description="Path to the video file")
    classifier_model_path: str = Field(..., description="Path to YOLO classifier model")
    save_keyframes: bool = Field(default=True, description="Whether to save keyframe images")
    save_classified: bool = Field(default=True, description="Whether to save classified images")
    frame_selection_method: FrameSelectionMethod = Field(
        default=FrameSelectionMethod.BY_REGION,
        description="Method for selecting frames"
    )
    num_select: int = Field(default=5, ge=1, le=50, description="Number of frames to select (non-by_region methods)")
    samples_per_region: int = Field(default=3, ge=1, le=10, description="Keyframes to sample per motion region")
    motion_threshold: float = Field(default=0.02, ge=0.001, le=0.5, description="Motion detection threshold")


class BatchProcessRequest(BaseModel):
    """Request to process multiple videos"""
    video_directory: str = Field(..., description="Directory containing video files")
    classifier_model_path: str = Field(..., description="Path to YOLO classifier model")
    video_extensions: List[str] = Field(default=[".mp4", ".avi", ".mov"], description="Video file extensions to process")
    frame_selection_method: FrameSelectionMethod = Field(default=FrameSelectionMethod.BY_REGION)
    num_select: int = Field(default=5, ge=1, le=50)


# Response Models
class MotionRegion(BaseModel):
    """Motion region detected in video"""
    region_id: int
    start_frame: int
    end_frame: int
    max_intensity: float
    duration_seconds: Optional[float] = None


class ClassificationResult(BaseModel):
    """Single frame classification result"""
    frame_idx: int
    timestamp: float
    region_id: int
    predicted_class: str
    class_id: int
    confidence: float
    all_probs: Dict[str, float]
    saved_path: Optional[str] = None


class VerificationResult(BaseModel):
    """VLM verification result"""
    frame_idx: int
    timestamp: float
    classifier_prediction: str
    classifier_confidence: float
    vlm_assessment: str
    vlm_confidence: float
    match: bool
    confidence_agreement: float
    vlm_reasoning: str
    verification_status: str


class DriftAnalysis(BaseModel):
    """Data drift analysis result"""
    drift_detected: bool
    drift_score: float
    mismatch_rate: float
    confidence_disagreement: float
    class_dominance: float
    predictions: Dict[str, int]
    total_samples: int
    mismatches: int


class RetrainingRecommendation(BaseModel):
    """Retraining recommendation"""
    timestamp: str
    recommendation: str
    confidence: float
    rationale: str
    actions: List[str]
    drift_score: float
    mismatch_rate: float
    verified_samples: int
    drift_detected: bool


class ProcessingSummary(BaseModel):
    """Summary of video processing results"""
    video_path: str
    motion_regions: int
    total_keyframes_sampled: int
    frames_selected: int
    frame_selection_method: str
    reduction_percentage: float


class VideoProcessResult(BaseModel):
    """Complete video processing result"""
    job_id: str
    status: JobStatus
    video_path: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    summary: Optional[ProcessingSummary] = None
    motion_regions: Optional[List[MotionRegion]] = None
    classifications: Optional[List[ClassificationResult]] = None
    verifications: Optional[List[VerificationResult]] = None
    drift_analysis: Optional[DriftAnalysis] = None
    recommendation: Optional[RetrainingRecommendation] = None
    error: Optional[str] = None


class JobInfo(BaseModel):
    """Basic job information"""
    job_id: str
    status: JobStatus
    video_path: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_phase: Optional[str] = None
    progress_percent: Optional[float] = None


class JobListResponse(BaseModel):
    """Response for listing jobs"""
    jobs: List[JobInfo]
    total: int


# WebSocket Event Models
class WebSocketEvent(BaseModel):
    """WebSocket event structure"""
    job_id: str
    event_type: str  # 'progress', 'phase_complete', 'error', 'complete'
    phase: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
