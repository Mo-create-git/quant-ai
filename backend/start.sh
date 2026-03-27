#!/bin/bash
cd ~/backend-QuantAI/backend
echo "🚀 Starting QUANT-AI..."
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
sleep 2
open http://localhost:8000
wait
