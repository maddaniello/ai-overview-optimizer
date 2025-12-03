#!/bin/bash

# Quick Start Script - Avvio rapido dopo clone

echo "🚀 AI Overview Optimizer - Quick Start"
echo "======================================"
echo ""

# Check se setup già fatto
if [ ! -d "venv" ]; then
    echo "📦 Prima installazione rilevata"
    echo "Eseguo setup completo..."
    echo ""
    chmod +x setup.sh
    ./setup.sh
else
    echo "✅ Environment già configurato"
fi

echo ""
echo "🔌 Attivazione environment..."
source venv/bin/activate

echo ""
echo "🚀 Avvio Streamlit..."
echo ""
streamlit run app.py