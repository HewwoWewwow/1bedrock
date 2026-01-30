#!/bin/bash
# Run the Streamlit UI for Austin TIF Model

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run streamlit
echo "Starting Austin TIF Model UI..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run ui/app.py --server.port 8501
