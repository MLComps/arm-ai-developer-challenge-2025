"""
Ambient Wildlife Monitoring API

FastAPI backend for video processing pipeline with real-time WebSocket updates.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from .config import get_settings
from .api.routes import videos, websocket
from .services.video_processor import init_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown"""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("AMBIENT WILDLIFE MONITORING API")
    logger.info("=" * 60)

    # Use absolute paths
    model_path = settings.absolute_classifier_model_path
    output_path = settings.absolute_output_dir

    # Validate paths
    if not os.path.exists(model_path):
        logger.warning(f"Classifier model not found: {model_path}")
        logger.warning("API will start but processing will fail until model is available")

    # Initialize video processor
    try:
        init_processor(
            classifier_model_path=model_path,
            output_base_dir=output_path,
            vlm_endpoint=settings.vlm_endpoint,
            vlm_model=settings.vlm_model,
            vlm_num_samples=settings.vlm_num_samples
        )
        logger.info(f"Video processor initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  VLM: {settings.vlm_endpoint} ({settings.vlm_model})")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")

    logger.info("=" * 60)
    logger.info(f"API ready at http://{settings.host}:{settings.port}")
    logger.info(f"Docs at http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Ambient Wildlife Monitoring API

This API provides video processing capabilities for wildlife monitoring using:

- **Motion Detection** (MOG2 background subtraction)
- **Keyframe Sampling** (intelligent frame extraction)
- **Classification** (YOLO-based animal detection)
- **VLM Verification** (Vision Language Model validation)
- **Drift Detection** (model performance monitoring)
- **Retraining Recommendations** (autonomous decisions)

### WebSocket Support

Connect to WebSocket endpoints for real-time processing updates:

- `/ws` - Global updates for all jobs
- `/api/videos/ws/{job_id}` - Updates for specific job

### Quick Start

1. POST `/api/videos/process` with video path to start processing
2. Connect to WebSocket for progress updates
3. GET `/api/videos/{job_id}/results` when complete
        """,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(videos.router, prefix="/api")
    app.include_router(websocket.router)

    # Get static directory path (relative to backend folder)
    static_dir = Path(__file__).parent.parent / "static"
    # Also check if running from project root
    if not static_dir.exists():
        static_dir = Path("backend/static")

    # Mount video assets directory for video preview
    from .config import get_project_root
    video_assets_dir = get_project_root() / "video-assets" / "normal"
    if video_assets_dir.exists():
        app.mount("/videos", StaticFiles(directory=str(video_assets_dir)), name="videos")

    # Mount output directory for classified images
    output_dir = Path(get_settings().absolute_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    # Mount static files if directory exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        """Serve the frontend UI"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs",
            "ui": "/static/index.html"
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy"}

    @app.get("/api")
    async def api_info():
        """API info"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs"
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
