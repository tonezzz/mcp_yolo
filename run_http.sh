#!/bin/bash

source /app/yolo-mcp-services/.venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
