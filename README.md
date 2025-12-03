# 🔍 AI Overview Content Optimizer

Strumento professionale per ottimizzare contenuti web e massimizzare le probabilità di comparire come fonte negli **AI Overview di Google**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sviluppato da [**Moca Interactive**](https://mocainteractive.com)

---

## ✨ Caratteristiche

- **Analisi SERP Completa**: Recupera AI Overview, fonti e query correlate
- **Web Scraping Avanzato**: Estrazione contenuti con Crawl4AI + BeautifulSoup fallback
- **Relevance Scoring**: Calcolo rilevanza contestuale con Jina Reranker o Google Vertex AI
- **Semantic Analysis**: Similarità semantica con OpenAI embeddings
- **Entity Gap Analysis**: Identifica entità mancanti vs competitor (NER con spaCy)
- **LLM Optimization**: Generazione versione ottimizzata con GPT-4o
- **Fan-out Opportunities**: Analisi query correlate per nuovi contenuti
- **UI Professionale**: Interfaccia Streamlit con branding Moca

---

## 🚀 Quick Start

### Installazione
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/ai-overview-optimizer.git
cd ai-overview-optimizer

# Setup automatico
chmod +x quick-start.sh
./quick-start.sh
```

### Configurazione

1. Copia `.env.example` in `.env`
2. Aggiungi le tue API keys:
```bash
DATAFORSEO_LOGIN=your_login
DATAFORSEO_PASSWORD=your_password
OPENAI_API_KEY=sk-your-key
JINA_API_KEY=jina_your-key
```

3. Avvia l'app:
```bash
streamlit run app.py
```

4. Apri browser su: `http://localhost:8501`

---

## 📋 Requisiti

- **Python**: 3.8 o superiore
- **API Keys**:
  - DataForSEO (SERP data)
  - OpenAI (embeddings + LLM)
  - Jina AI (reranking) *oppure* Google Vertex AI

---

## 💰 Costi Stimati

**Per singola analisi**: ~$0.15-0.30

- DataForSEO: ~$0.10/query
- OpenAI embeddings: ~$0.001
- Jina: tier gratuito disponibile

---

## 📊 Output

L'analisi fornisce:

1. **Relevance Score**: punteggio rilevanza contestuale (0-1)
2. **Top Sources**: fonti AI Overview con scores comparativi
3. **Entity Gap**: entità mancanti nel tuo contenuto
4. **Optimized Answer**: versione migliorata del contenuto
5. **Fan-out Queries**: opportunità per nuove pagine
6. **Recommendations**: suggerimenti actionable

---

## 🛠️ Stack Tecnologico

- **Backend**: Python 3.11, FastAPI
- **Frontend**: Streamlit
- **Web Scraping**: Crawl4AI, BeautifulSoup4
- **APIs**: DataForSEO, OpenAI, Jina AI
- **NLP**: spaCy, scikit-learn
- **Data**: Pandas, NumPy

---

## 📖 Documentazione

- [Usage Guide](docs/USAGE_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Contributing](docs/CONTRIBUTING.md)

---

## ⚠️ Limitazioni

- **Non garantisce** inclusione in AI Overview (migliora solo rilevanza)
- Keywords transactional raramente hanno AI Overview
- Rate limits API da rispettare
- Costi variabili in base a numero analisi

---

## 🤝 Contributing

Contributi benvenuti! Leggi [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## 📄 License

MIT License - vedi [LICENSE](LICENSE)

---

## 🏢 About Moca Interactive

Sviluppato da [**Moca Interactive**](https://mocainteractive.com) - Digital Marketing Agency

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR-USERNAME/ai-overview-optimizer/issues)
- **Email**: support@mocainteractive.com
- **Website**: [mocainteractive.com](https://mocainteractive.com)

---

⭐ Se questo progetto ti è utile, lascia una star su GitHub!