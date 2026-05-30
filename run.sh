#!/bin/bash
# FusionNet-Scratch Launcher Script

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/.venv"

echo "🔧 FusionNet-Scratch Web Application Launcher"
echo "=============================================="

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install requirements
echo "📦 Installing/updating dependencies..."
pip install -q -r "$PROJECT_DIR/requirements.txt"

# Start server
echo ""
echo "✅ Environment ready!"
echo ""
echo "🚀 Starting FusionNet-Scratch Web Server..."
echo "   URL: http://localhost:8001"
echo "   API Docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$PROJECT_DIR"
python -m uvicorn webapp.backend:app --host 0.0.0.0 --port 8001 --reload
