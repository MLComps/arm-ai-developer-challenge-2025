# Ambient Wildlife Monitoring – ARM AI Developer Challenge 2025

A sophisticated, self-aware edge AI system for autonomous wildlife monitoring that runs continuously on ARM-based devices without cloud dependency. Built for the ARM AI Developer Challenge 2025.

Demo: [Video 1](https://www.youtube.com/watch?v=kxtysijYQKw) | [Video 2](https://www.youtube.com/watch?v=gR7cQX7ev9g) | [Video 3](https://www.youtube.com/shorts/ZV12m5BS5pM)
Demo: [Video 1](https://www.youtube.com/watch?v=kxtysijYQKw) | [Video 2](https://www.youtube.com/watch?v=gR7cQX7ev9g) | [Video 3](https://www.youtube.com/shorts/ZV12m5BS5pM) | [[Video 4]](https://www.youtube.com/watch?v=064JLW0dzmU)

- Dataset - [Link](https://www.kaggle.com/datasets/silviamatoke/serengeti-dataset/data)
- More about Serengeti Dataset - [Link](https://lila.science/datasets/snapshot-serengeti/)

## Overview

Ambient Wildlife Guard demonstrates enterprise-grade edge AI capabilities by deploying a complete wildlife classification pipeline on resource-constrained ARM devices. The system processes camera trap footage through an intelligent 6-phase pipeline optimized for edge deployment.

## Model Performance Comparison

### Untrained Model (Baseline)

Random initialization with zero training on wildlife data.

![Untrained Classification Results](./assets/2.png)

### Trained Model (50 Epochs)

Fine-tuned on 225 training images after optimization.

![Trained Classification Results](./assets/1.png)

### System Recommendations

Autonomous decision-making with confidence thresholds and VLM verification.

![Recommended Actions](./assets/3.png)

## Core Features

**6-Phase Intelligent Pipeline:**

1. **Motion Detection** - MOG2 background subtraction optimized for static camera trap scenarios
2. **Intelligent Keyframe Selection** - Reduces 500 frames to 9 strategically selected frames (98% reduction)
3. **Multi-Class Classification** - YOLOv8-based classification system.
4. **VLM Verification** - Cross-validation using Vision Language Models (qwen3-vl:2b)
5. **Data Drift Detection** - Monitors classification confidence mismatches to identify distribution shifts
6. **Autonomous Retraining** - Recommends model updates when performance degrades

### Architecture

A modular, extensible wildlife monitoring pipeline designed for edge devices.
Supports motion detection → keyframe sampling → image classification → VLM verification → data drift detection.

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

## VLM Verification Flow

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

## Prerequisites

- Python 3.11+
- uv (Python virtual environment)
- [Ollama](https://ollama.ai) (for VLM verification)
- ARM-based device (or compatible development machine)
- 8GB+ RAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/MLComps/arm-ai-developer-challenge-2025.git
cd arm-ai-developer-challenge-2025

# Install dependencies using uv (fast Python package manager)
uv sync

# Activate virtual environment
source ./.venv/bin/activate

# Install backend requirements
cd backend
uv pip install -r requirements.txt
```

### Setup Ollama VLM

The system uses Ollama for Vision Language Model verification. Download and install Ollama, then pull a model:

```bash
# Install Ollama from https://ollama.ai
# Pull a lightweight vision model
ollama pull qwen3-vl:2b

# Start Ollama server (runs on localhost:11434 by default)
ollama serve
```

### Running the System

```bash
# From the backend directory
python run.py
```

You should see startup output confirming all components:

```
============================================================
AMBIENT WILDLIFE MONITORING API
============================================================
VLM endpoint configured: http://localhost:11434/api/generate with model qwen3-vl:2b
Video processor initialized
  Model: ./classifier-models/train-fox-background/weights/best.pt
  Output: ./output
  VLM: http://localhost:11434/api/generate (qwen3-vl:2b)
============================================================
API ready at http://0.0.0.0:8000
Docs at http://0.0.0.0:8000/docs
============================================================
```

Access the API and interactive documentation at `http://localhost:8000`

### What's next for Ambient Agent for Wildlife Monitoring on Edge

- Explore executorch for Ultralytics
- Human-in-the-loop labelling
- Animal activity tracker
