# Ambient Wildlife Monitoring System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AMBIENT WILDLIFE MONITORING                            │
│                              System Architecture                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     WebSocket      ┌──────────────────────────────────────────────┐
│              │◄──────────────────►│                                              │
│   Frontend   │                    │              FastAPI Backend                 │
│   (HTML/JS)  │    REST API        │                                              │
│              │◄──────────────────►│                                              │
└──────────────┘                    └──────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                    ┌──────────────────────────────────────────────┐
                                    │           Video Processing Pipeline          │
                                    │                                              │
                                    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
                                    │  │ Motion  │─►│Keyframe │─►│   Frame     │  │
                                    │  │Detection│  │Sampling │  │  Selection  │  │
                                    │  └─────────┘  └─────────┘  └─────────────┘  │
                                    │       │            │              │          │
                                    │       ▼            ▼              ▼          │
                                    │  ┌─────────────────────────────────────────┐ │
                                    │  │         YOLO Classification             │ │
                                    │  │      (Dynamic Model Selection)          │ │
                                    │  └─────────────────────────────────────────┘ │
                                    │                      │                       │
                                    │                      ▼                       │
                                    │  ┌─────────────────────────────────────────┐ │
                                    │  │      VLM Verification (Ollama)          │ │
                                    │  │         qwen3-vl:2b Model               │ │
                                    │  └─────────────────────────────────────────┘ │
                                    │                      │                       │
                                    │       ┌──────────────┴──────────────┐        │
                                    │       ▼                             ▼        │
                                    │  ┌──────────┐              ┌──────────────┐  │
                                    │  │  Drift   │              │  Retraining  │  │
                                    │  │Detection │─────────────►│Recommendation│  │
                                    │  └──────────┘              └──────────────┘  │
                                    └──────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend (Web UI)

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (HTML/JS)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Video Selection   │    │      Model Selection            │ │
│  │   - Video dropdown  │    │   - Dynamic model list          │ │
│  │   - Video preview   │    │   - Class tags display          │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Classifier Predictions Grid                  ││
│  │   - 9 classified frames with images                         ││
│  │   - Class labels + confidence scores                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  VLM Verification Grid                       ││
│  │   - Green overlay = Valid classification                    ││
│  │   - Red overlay = Invalid classification                    ││
│  │   - VLM observations displayed                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Summary Panel     │    │      Activity Log Sidebar       │ │
│  │   - Recommendation  │    │   - Real-time streaming logs    │ │
│  │   - Metrics         │    │   - Phase progress tracking     │ │
│  │   - Actions         │    │   - Progress bar                │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Backend Services

```
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      API Routes                              ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  POST /api/videos/process     - Start video processing      ││
│  │  POST /api/videos/batch       - Batch process directory     ││
│  │  GET  /api/videos/models      - List available models       ││
│  │  GET  /api/videos/{id}        - Get job details             ││
│  │  GET  /api/videos/{id}/status - Get job status              ││
│  │  GET  /api/videos/{id}/results- Get job results             ││
│  │  WS   /ws                     - Global WebSocket updates    ││
│  │  WS   /api/videos/ws/{id}     - Job-specific WebSocket      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Static File Serving                        ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  /              - Frontend UI (index.html)                  ││
│  │  /static/       - Static assets                             ││
│  │  /videos/       - Video assets for preview                  ││
│  │  /output/       - Classified images output                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  WebSocket Manager                           ││
│  │   - Connection management                                   ││
│  │   - Job-specific subscriptions                              ││
│  │   - Real-time event broadcasting                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline

### Phase Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO PROCESSING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────┐
     │  Video  │
     │  Input  │
     └────┬────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 1: Motion    │     ┌──────────────────────────────────────────┐
│     Detection       │────►│ - MOG2 Background Subtraction            │
│                     │     │ - Identifies frames with movement        │
│   (MOG2 Algorithm)  │     │ - Groups into motion regions             │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 2: Keyframe  │     ┌──────────────────────────────────────────┐
│     Sampling        │────►│ - Samples frames from each motion region │
│                     │     │ - Configurable samples per region        │
│                     │     │ - Saves keyframe images                  │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ PHASE 2.5: Frame    │     ┌──────────────────────────────────────────┐
│    Selection        │────►│ - Balanced selection method              │
│                     │     │ - Selects 9 diverse frames               │
│ (Diversity-based)   │     │ - Reduces redundancy                     │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 3: YOLO      │     ┌──────────────────────────────────────────┐
│   Classification    │────►│ - Dynamic model selection                │
│                     │     │ - Classes: background, fox, deer, etc.   │
│  (Dynamic Models)   │     │ - Confidence scores per class            │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 4: VLM       │     ┌──────────────────────────────────────────┐
│   Verification      │────►│ - Samples 5 lowest-confidence frames     │
│                     │     │ - Ollama qwen3-vl:2b model               │
│  (Ollama VLM)       │     │ - Returns: class_valid + observation     │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 5: Drift     │     ┌──────────────────────────────────────────┐
│    Detection        │────►│ - Analyzes mismatch patterns             │
│                     │     │ - Calculates drift score                 │
│                     │     │ - Monitors model degradation             │
└─────────┬───────────┘     └──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 6: Retrain   │     ┌──────────────────────────────────────────┐
│   Recommendation    │────►│ - Aggregate confidence calculation       │
│                     │     │ - Threshold: 0.7 (70%)                   │
│                     │     │ - Actions: URGENT_RETRAIN, RETRAIN,      │
│                     │     │            MONITOR_CLOSELY, CONTINUE     │
└─────────────────────┘     └──────────────────────────────────────────┘
```

---

## Data Flow

### Request/Response Flow

```
┌──────────┐                    ┌──────────┐                    ┌──────────────┐
│  Client  │                    │  FastAPI │                    │  Pipeline    │
│ (Browser)│                    │  Server  │                    │  Processor   │
└────┬─────┘                    └────┬─────┘                    └──────┬───────┘
     │                               │                                  │
     │  1. POST /api/videos/process  │                                  │
     │──────────────────────────────►│                                  │
     │                               │                                  │
     │  2. Return job_id             │                                  │
     │◄──────────────────────────────│                                  │
     │                               │                                  │
     │  3. Connect WebSocket /ws     │                                  │
     │──────────────────────────────►│                                  │
     │                               │  4. Start background processing  │
     │                               │─────────────────────────────────►│
     │                               │                                  │
     │                               │  5. Phase updates (via callback) │
     │  6. WebSocket events          │◄─────────────────────────────────│
     │◄──────────────────────────────│                                  │
     │  (real-time progress)         │                                  │
     │                               │                                  │
     │  7. job_completed event       │  8. Processing complete          │
     │◄──────────────────────────────│◄─────────────────────────────────│
     │                               │                                  │
     │  9. GET /api/videos/{id}/results                                 │
     │──────────────────────────────►│                                  │
     │                               │                                  │
     │  10. Full results JSON        │                                  │
     │◄──────────────────────────────│                                  │
     │                               │                                  │
```

### VLM Verification Flow

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Classified     │         │   VLM Verifier   │         │  Ollama Server   │
│     Frames       │         │                  │         │  (qwen3-vl:2b)   │
└────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
         │                            │                            │
         │  1. 9 classified frames    │                            │
         │───────────────────────────►│                            │
         │                            │                            │
         │                            │  2. Select 5 frames        │
         │                            │     (3 lowest conf +       │
         │                            │      2 random)             │
         │                            │                            │
         │                            │  3. For each frame:        │
         │                            │     POST /api/generate     │
         │                            │────────────────────────────►
         │                            │     {image, prompt}        │
         │                            │                            │
         │                            │  4. VLM Response           │
         │                            │◄────────────────────────────
         │                            │     VALID: YES/NO          │
         │                            │     OBSERVATION: ...       │
         │                            │                            │
         │  5. Verification results   │                            │
         │◄───────────────────────────│                            │
         │     - class_valid          │                            │
         │     - observation          │                            │
         │     - validity_rate        │                            │
```

---

## Directory Structure

```
arm-ai-developer-challenge-2025/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration settings
│   │   │
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── videos.py       # Video processing endpoints
│   │   │       └── websocket.py    # WebSocket handlers
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic models
│   │   │
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── motion_detector.py      # Phase 1
│   │   │   ├── keyframe_sampler.py     # Phase 2
│   │   │   ├── frame_selector.py       # Phase 2.5
│   │   │   ├── classifier.py           # Phase 3
│   │   │   ├── vlm_verifier.py         # Phase 4
│   │   │   ├── drift_detector.py       # Phase 5
│   │   │   └── retraining_recommender.py # Phase 6
│   │   │
│   │   └── services/
│   │       ├── video_processor.py  # Pipeline orchestrator
│   │       └── websocket_manager.py
│   │
│   ├── static/
│   │   └── index.html              # Frontend UI
│   │
│   └── run.py                      # Server entry point
│
├── classifier-models/
│   └── {model-name}/
│       └── weights/
│           └── best.pt             # YOLO model weights
│
├── video-assets/
│   └── normal/
│       └── *.mp4                   # Input videos
│
└── output/
    └── job_{id}_{video}/
        ├── keyframes/              # Extracted keyframes
        └── classified/
            └── {class}/            # Classified images by class
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                       TECHNOLOGY STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frontend          │  Backend           │  ML/AI                │
│  ─────────────────────────────────────────────────────────────  │
│  • HTML5           │  • Python 3.11     │  • YOLO (Ultralytics) │
│  • CSS3            │  • FastAPI         │  • OpenCV             │
│  • JavaScript      │  • Uvicorn         │  • NumPy              │
│  • WebSocket API   │  • Pydantic        │  • Ollama             │
│                    │  • asyncio         │  • qwen3-vl:2b        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Communication                    │  Storage                    │
│  ────────────────────────────────────────────────────────────   │
│  • REST API                       │  • File system (videos)     │
│  • WebSocket (real-time)          │  • File system (models)     │
│  • JSON data format               │  • File system (output)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommendation Logic

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION DECISION TREE                  │
└─────────────────────────────────────────────────────────────────┘

                    Calculate Aggregate Confidence
                    ════════════════════════════════
                    aggregate_conf = avg_classifier_conf × vlm_validity_rate

                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  aggregate_conf < 0.7 (70%)? │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                   YES                              NO
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ vlm_validity_rate │           │ aggregate_conf    │
        │    < 0.5 (50%)?   │           │    > 0.85?        │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
        ┌─────────┴─────────┐           ┌─────────┴─────────┐
        │                   │           │                   │
       YES                  NO         YES                  NO
        │                   │           │                   │
        ▼                   ▼           ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    URGENT     │  │   RETRAIN     │  │   CONTINUE    │  │    MONITOR    │
│    RETRAIN    │  │ RECOMMENDED   │  │  MONITORING   │  │    CLOSELY    │
│               │  │               │  │               │  │               │
│ Stop using    │  │ Schedule      │  │ No action     │  │ Increase      │
│ model in      │  │ retraining    │  │ required      │  │ monitoring    │
│ production    │  │               │  │               │  │ frequency     │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

---

## WebSocket Events

```
┌─────────────────────────────────────────────────────────────────┐
│                      WEBSOCKET EVENT TYPES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Event Type              │  Payload                              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  job_started             │  { job_id, video_path }               │
│                                                                  │
│  motion_detection        │  { status, progress_percent,          │
│                          │    motion_regions, ... }              │
│                                                                  │
│  keyframe_sampling       │  { status, progress_percent,          │
│                          │    keyframes_sampled, ... }           │
│                                                                  │
│  frame_selection         │  { status, selected_frames,           │
│                          │    reduction_percent }                │
│                                                                  │
│  classification          │  { status, progress_percent,          │
│                          │    last_prediction: {                 │
│                          │      frame_idx, class,                │
│                          │      confidence, saved_path           │
│                          │    }}                                 │
│                                                                  │
│  vlm_verification        │  { status, progress_percent,          │
│                          │    valid_count, invalid_count,        │
│                          │    last_verification: {               │
│                          │      frame_idx, classifier,           │
│                          │      class_valid, observation,        │
│                          │      saved_path                       │
│                          │    }}                                 │
│                                                                  │
│  drift_detection         │  { status, drift_detected,            │
│                          │    drift_score, mismatch_rate }       │
│                                                                  │
│  retraining_recommendation│ { status, recommendation,            │
│                          │    rationale, actions, metrics }      │
│                                                                  │
│  job_completed           │  { job_id, summary,                   │
│                          │    recommendation }                   │
│                                                                  │
│  job_failed              │  { job_id, error }                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## External Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Ollama Server                         │    │
│  │                                                          │    │
│  │  Endpoint: http://localhost:11434/api/generate           │    │
│  │  Model: qwen3-vl:2b (Vision Language Model)              │    │
│  │                                                          │    │
│  │  Purpose: Verify classifier predictions by analyzing     │    │
│  │           images and providing observations              │    │
│  │                                                          │    │
│  │  Input: Base64 encoded image + verification prompt       │    │
│  │  Output: VALID: YES/NO + OBSERVATION text                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```yaml
# Default Configuration (backend/app/config.py)

API Settings:
  app_name: "Ambient Wildlife Monitoring API"
  app_version: "1.0.0"
  host: "0.0.0.0"
  port: 8000

Paths:
  classifier_models: "classifier-models/"
  video_assets: "video-assets/normal/"
  output: "output/"

VLM Settings:
  endpoint: "http://localhost:11434/api/generate"
  model: "qwen3-vl:2b"
  num_samples: 5  # Frames to verify

Processing Defaults:
  frame_selection_method: "balanced"
  num_select: 9
  samples_per_region: 5
  motion_threshold: 0.02

Recommendation:
  confidence_threshold: 0.7  # 70%
  mismatch_threshold: 0.3    # 30%
```
