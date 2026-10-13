#!/bin/bash
set -e

# Always run from the project root (where this script lives)
cd "$(dirname "$0")"

echo "Starting FastAPI backend on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 15); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" 2>/dev/null; then
        echo "FastAPI is ready."
        break
    fi
    sleep 1
done

echo "Starting Flask frontend on port 5000..."
FASTAPI_URL=http://localhost:8000 python frontend/app.py &
FLASK_PID=$!

echo ""
echo "Both services running:"
echo "  Web UI  → http://localhost:5000"
echo "  API     → http://localhost:8000"
echo "  API docs→ http://localhost:8000/docs"
echo ""

# Wait for either process to exit
wait $FASTAPI_PID $FLASK_PID
