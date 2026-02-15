#!/bin/bash
# Quick Start Script for Merchant Fraud Detection Simulator
# Run this to set up and launch the system in one command

set -e  # Exit on error

echo "=========================================="
echo "Merchant Fraud Detection Simulator"
echo "Quick Start Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$python_version" "$required_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Create directories
echo "Creating project directories..."
mkdir -p data/raw data/processed models logs reports
echo "✅ Directories created"

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short 2>/dev/null || python tests/test_fraud_detection.py
echo "✅ Tests complete"

# Launch the application
echo ""
echo "=========================================="
echo "🚀 Launching Merchant Fraud Detection Simulator"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Generate synthetic merchant data (10K merchants)"
echo "  2. Generate transaction data (500K transactions)"
echo "  3. Run risk scoring engine"
echo "  4. Train ML anomaly detection model"
echo "  5. Create investigation cases"
echo "  6. Launch interactive dashboard"
echo ""
echo "Access the dashboard at: http://localhost:8501"
echo ""
echo "Press Ctrl+C at any time to stop"
echo ""
read -p "Press Enter to continue..."

# Run main pipeline
python main.py

# Deactivate on exit
deactivate
