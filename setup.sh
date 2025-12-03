#!/bin/bash

# AI Overview Content Optimizer - Setup Script
# Configura ambiente e dipendenze

echo "🔍 AI Overview Content Optimizer - Setup"
echo "========================================"
echo ""

# Controlla Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trovato"
    echo "Installa Python 3.8+ da https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION trovato"
echo ""

# Crea virtual environment
echo "📦 Creazione virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment creato"
else
    echo "❌ Errore creazione venv"
    exit 1
fi

# Attiva venv
echo ""
echo "🔌 Attivazione environment..."
source venv/bin/activate

# Aggiorna pip
echo ""
echo "⬆️  Aggiornamento pip..."
pip install --upgrade pip setuptools wheel

# Installa dipendenze
echo ""
echo "📥 Installazione dipendenze..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dipendenze installate"
else
    echo "❌ Errore installazione dipendenze"
    exit 1
fi

# Setup Crawl4AI
echo ""
echo "🕷️  Setup Crawl4AI..."
crawl4ai-setup

# Download spaCy models
echo ""
echo "📚 Download modelli spaCy..."
python3 -m spacy download it_core_news_lg
python3 -m spacy download en_core_web_lg

# Crea .env se non esiste
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creazione file .env..."
    cp .env.example .env
    echo "⚠️  Configura le API keys in .env"
fi

# Crea directories
mkdir -p logs .cache

echo ""
echo "========================================"
echo "✅ Setup completato!"
echo "========================================"
echo ""
echo "📝 Prossimi passi:"
echo ""
echo "1. Configura API keys:"
echo "   nano .env"
echo ""
echo "2. Attiva environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Avvia app:"
echo "   streamlit run app.py"
echo ""
echo "🌐 App disponibile su: http://localhost:8501"
echo ""