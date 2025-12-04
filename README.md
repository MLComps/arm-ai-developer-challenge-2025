# arm-ai-developer-challenge-2025
The repo contains the codebase for the ARM Developer Edge AI  competition.


```
uv sync 
```

```
source ./.venv/bin/activate    
```

```
cd backend 
```

```
uv pip install -r requirements.txt
```

```
python run.py
```

```
INFO:     Waiting for application startup.
2025-12-04 11:32:03,444 - app.main - INFO - ============================================================
2025-12-04 11:32:03,444 - app.main - INFO - AMBIENT WILDLIFE MONITORING API
2025-12-04 11:32:03,444 - app.main - INFO - ============================================================
2025-12-04 11:32:03,444 - app.pipeline.vlm_verifier - INFO - VLM endpoint configured: http://localhost:11434/api/generate with model qwen3-vl:2b
2025-12-04 11:32:03,444 - app.main - INFO - Video processor initialized
2025-12-04 11:32:03,444 - app.main - INFO -   Model: /Users/george/Documents/github/mlcomps/v2/arm-ai-developer-challenge-2025/classifier-models/train-fox-background/weights/best.pt
2025-12-04 11:32:03,444 - app.main - INFO -   Output: /Users/george/Documents/github/mlcomps/v2/arm-ai-developer-challenge-2025/output
2025-12-04 11:32:03,444 - app.main - INFO -   VLM: http://localhost:11434/api/generate (qwen3-vl:2b)
2025-12-04 11:32:03,444 - app.main - INFO - ============================================================
2025-12-04 11:32:03,444 - app.main - INFO - API ready at http://0.0.0.0:8000
2025-12-04 11:32:03,444 - app.main - INFO - Docs at http://0.0.0.0:8000/docs
2025-12-04 11:32:03,444 - app.main - INFO - ============================================================
```

Visit the link below on your local browser: 
```
http://0.0.0.0:8000

```
# TODO 
- Explore executorch for Ultralytics
- Human detection system 
- Animal activity tracker
- Ollama installation instruction 
- System Arch.