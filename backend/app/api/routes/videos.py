"""
Video Processing API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Optional
import os
from pathlib import Path

from ...models.schemas import (
    VideoProcessRequest,
    BatchProcessRequest,
    JobInfo,
    JobListResponse,
    JobStatus
)
from ...services.video_processor import get_processor
from ...services.websocket_manager import manager
from ...config import get_project_root

router = APIRouter(prefix="/videos", tags=["videos"])


def resolve_path(path: str) -> str:
    """Resolve relative path to absolute using project root"""
    if os.path.isabs(path):
        return path
    project_root = get_project_root()
    return str(project_root / path)


@router.post("/process", response_model=JobInfo)
async def process_video(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Queue a video for processing.

    Returns immediately with job ID. Connect to WebSocket for progress updates.
    """
    # Resolve paths to absolute
    video_path = resolve_path(request.video_path)
    model_path = resolve_path(request.classifier_model_path)

    # Validate video path exists
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_path}"
        )

    # Validate model path exists
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_path}"
        )

    processor = get_processor()

    # Create job synchronously to get job_id immediately
    job_id = processor._create_job(video_path)
    job = processor.get_job(job_id)

    async def emit_to_websocket(event: dict):
        """Emit events to WebSocket connections"""
        await manager.send_to_job(job_id, event)

    # Run processing in background
    async def run_processing():
        # Re-initialize the job (it was created above)
        processor.jobs[job_id]["status"] = JobStatus.PENDING

        await processor.process_video(
            video_path=video_path,
            on_event=emit_to_websocket,
            frame_selection_method=request.frame_selection_method,
            num_select=request.num_select,
            samples_per_region=request.samples_per_region,
            motion_threshold=request.motion_threshold,
            save_keyframes=request.save_keyframes,
            save_classified=request.save_classified
        )

    background_tasks.add_task(run_processing)

    return JobInfo(
        job_id=job_id,
        status=job["status"],
        video_path=job["video_path"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        current_phase=job["current_phase"],
        progress_percent=job["progress_percent"]
    )


@router.post("/batch", response_model=List[JobInfo])
async def batch_process(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Queue all videos in a directory for processing.
    """
    # Resolve path to absolute
    video_dir = resolve_path(request.video_directory)

    # Validate directory exists
    if not os.path.isdir(video_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {video_dir}"
        )

    # Find video files
    video_files = []
    for ext in request.video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))

    if not video_files:
        raise HTTPException(
            status_code=404,
            detail=f"No video files found in {video_dir}"
        )

    processor = get_processor()
    jobs = []

    for video_path in sorted(video_files):
        video_path_str = str(video_path)
        job_id = processor._create_job(video_path_str)
        job = processor.get_job(job_id)

        async def emit_to_websocket(event: dict, jid=job_id):
            await manager.send_to_job(jid, event)

        async def run_processing(vpath=video_path_str, jid=job_id, emit=emit_to_websocket):
            processor.jobs[jid]["status"] = JobStatus.PENDING
            await processor.process_video(
                video_path=vpath,
                on_event=emit,
                frame_selection_method=request.frame_selection_method,
                num_select=request.num_select
            )

        background_tasks.add_task(run_processing)

        jobs.append(JobInfo(
            job_id=job_id,
            status=job["status"],
            video_path=job["video_path"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            current_phase=job["current_phase"],
            progress_percent=job["progress_percent"]
        ))

    return jobs


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List all processing jobs.
    """
    processor = get_processor()
    all_jobs = processor.get_all_jobs()

    # Filter by status if specified
    if status:
        all_jobs = [j for j in all_jobs if j["status"] == status]

    total = len(all_jobs)

    # Apply pagination
    all_jobs = all_jobs[offset:offset + limit]

    job_infos = [
        JobInfo(
            job_id=j["job_id"],
            status=j["status"],
            video_path=j["video_path"],
            started_at=j["started_at"],
            completed_at=j["completed_at"],
            current_phase=j["current_phase"],
            progress_percent=j["progress_percent"]
        )
        for j in all_jobs
    ]

    return JobListResponse(jobs=job_infos, total=total)


@router.get("/{job_id}")
async def get_job(job_id: str):
    """
    Get job details and results.
    """
    processor = get_processor()
    job = processor.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@router.get("/{job_id}/status", response_model=JobInfo)
async def get_job_status(job_id: str):
    """
    Get job status (lightweight).
    """
    processor = get_processor()
    job = processor.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobInfo(
        job_id=job["job_id"],
        status=job["status"],
        video_path=job["video_path"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        current_phase=job["current_phase"],
        progress_percent=job["progress_percent"]
    )


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get complete job results.
    """
    processor = get_processor()
    job = processor.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] == JobStatus.PENDING:
        raise HTTPException(status_code=202, detail="Job is pending")

    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job is still processing")

    if job["status"] == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")

    return job["results"]


# WebSocket endpoint for job-specific updates
@router.websocket("/ws/{job_id}")
async def websocket_job(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.

    Connect to receive events for a specific job.
    """
    await manager.connect(websocket, job_id)

    try:
        # Send current job state if it exists
        processor = get_processor()
        job = processor.get_job(job_id)

        if job:
            await websocket.send_json({
                "event_type": "connected",
                "job_id": job_id,
                "current_status": job["status"].value,
                "current_phase": job["current_phase"],
                "progress_percent": job["progress_percent"]
            })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back for ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        await manager.disconnect(websocket, job_id)
