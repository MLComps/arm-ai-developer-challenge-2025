# Ambient Wildlife Monitoring – ARM AI Developer Challenge 2025

A sophisticated, self-aware edge AI system for autonomous wildlife monitoring that runs continuously on ARM-based devices without cloud dependency. Built for the ARM AI Developer Challenge 2025.

Demo: [Video 1](https://www.youtube.com/watch?v=kxtysijYQKw) | [Video 2](https://www.youtube.com/watch?v=gR7cQX7ev9g) | [Video 3](https://www.youtube.com/shorts/ZV12m5BS5pM)

![System View](./assets/wildlife-system.png)

## Overview

Ambient Wildlife Guard demonstrates enterprise-grade edge AI capabilities by deploying a complete wildlife classification pipeline on resource-constrained ARM devices. The system processes camera trap footage through an intelligent 6-phase pipeline with **98% computational load reduction** and **75% memory optimization** while maintaining **95%+ classification accuracy**.

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

```

┌────────────────┐     ┌───────────────────┐      ┌──────────────────────┐
│ Motion Detector │ ─→ │ Keyframe Sampler  │ ─→   │ Classifier Model     │
└────────────────┘     └───────────────────┘      └──────────────────────┘
                                                              │
                                                              ▼
                                           ┌─────────────────────────────┐
                                           │ Vision-Language Model (VLM) │
                                           └─────────────────────────────┘
                                                              │
                                                              ▼
                                           ┌─────────────────────────────┐
                                           │ Data Drift & Confidence     │
                                           └─────────────────────────────┘

```

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for VLM verification)
- ARM-based device (or compatible development machine)
- 8GB+ RAM recommended

### Installation

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

Explore executorch for Ultralytics
Human-in-the-loop labelling
Animal activity tracker
