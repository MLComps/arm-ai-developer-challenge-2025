#!/usr/bin/env python3
"""
Run the Ambient Wildlife Monitoring API server
"""

import uvicorn
import os

# Get the project root (parent of backend)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if __name__ == "__main__":
    # Change to BACKEND directory so 'app' module is found
    os.chdir(BACKEND_DIR)

    # But set env var so app knows project root for file paths
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Working directory: {os.getcwd()}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"]
    )
