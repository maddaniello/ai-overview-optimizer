"""
AI Overview Content Optimizer v2.0 - Multi-Agent System
Developed by Moca Interactive
"""
import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.base_agent import AgentState
from agents.orchestrator import OrchestratorAgent
from utils.logger import logger
from config import (
    MOCA_COLORS, MOCA_LOGO_URL, ALL_MODELS, OPENAI_MODELS, GEMINI_MODELS,
    LOCATION_CODES, LANGUAGE_CODES, DEFAULT_ITERATIONS, MAX_ITERATIONS,
    DEFAULT_SERP_RESULTS, MAX_SERP_RESULTS, MAX_SOURCES
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Overview Optimizer v2.0 | Moca",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'running' not in st.session_state:
    st.session_state.running = False

# ==================== CUSTOM CSS ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Figtree', sans-serif;
    }}

    .main {{
        background-color: #FFFFFF;
    }}

    h1, h2, h3 {{
        color: {MOCA_COLORS['primary']} !important;
        font-weight: 700;
    }}

    .stButton>button {{
        background-color: {MOCA_COLORS['primary']};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
    }}

    .stButton>button:hover {{
        background-color: #c41d13;
    }}

    .log-container {{
        background-color: #1a1a2e;
        color: #16c784;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.75rem;
        padding: 10px;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
    }}

    .log-info {{ color: #3b82f6; }}
    .log-success {{ color: #10b981; }}
    .log-warning {{ color: #f59e0b; }}
    .log-error {{ color: #ef4444; }}

    .iteration-card {{
        background: linear-gradient(135deg, {MOCA_COLORS['secondary']} 0%, #fff 100%);
        border-left: 4px solid {MOCA_COLORS['primary']};
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }}

    .ranking-table {{
        width: 100%;
        border-collapse: collapse;
    }}

    .ranking-table th {{
        background-color: {MOCA_COLORS['primary']};
        color: white;
        padding: 10px;
        text-align: left;
    }}

    .ranking-table td {{
        padding: 8px;
        border-bottom: 1px solid #eee;
    }}

    .ranking-table tr:hover {{
        background-color: {MOCA_COLORS['secondary']};
    }}

    .metric-card {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {MOCA_COLORS['primary']};
    }}

    .metric-label {{
        font-size: 0.9rem;
        color: {MOCA_COLORS['gray']};
    }}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def add_log(message: str, level: str = "info", agent: str = "system"):
    """Aggiunge un log alla sessione"""
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "level": level,
        "message": message
    }
    st.session_state.logs.append(log_entry)


def log_callback(log_entry: Dict):
    """Callback per ricevere log dagli agenti"""
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": log_entry.get("agent", "agent"),
        "level": log_entry.get("level", "info"),
        "message": log_entry.get("message", "")
    })


def render_logs():
    """Renderizza i log in formato console"""
    if not st.session_state.logs:
        return ""

    html_lines = []
    for log in st.session_state.logs[-50:]:  # Ultimi 50 log
        level_class = f"log-{log['level']}"
        html_lines.append(
            f'<span style="color:#666">[{log["timestamp"]}]</span> '
            f'<span class="{level_class}">[{log["agent"].upper()}]</span> '
            f'{log["message"]}'
        )

    return "<br>".join(html_lines)


async def run_optimization(state: AgentState):
    """Esegue il workflow di ottimizzazione"""
    orchestrator = OrchestratorAgent(log_callback=log_callback)
    result_state = await orchestrator.run(state)
    return result_state


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image(MOCA_LOGO_URL, width=80)
    st.markdown("# 🚀 AI Overview Optimizer")
    st.markdown("**v2.0 - Multi-Agent System**")

    st.divider()

    # ===== API KEYS =====
    with st.expander("🔑 API Keys", expanded=True):
        st.markdown("### DataForSEO")
        dataforseo_login = st.text_input(
            "Login",
            type="default",
            help="Il tuo login DataForSEO",
            key="dataforseo_login"
        )
        dataforseo_password = st.text_input(
            "Password",
            type="password",
            help="La tua password DataForSEO",
            key="dataforseo_password"
        )

        st.markdown("---")
        st.markdown("### OpenAI")
        openai_key = st.text_input(
            "API Key",
            type="password",
            help="Obbligatoria per embeddings. Formato: sk-...",
            key="openai_key"
        )

        st.markdown("---")
        st.markdown("### Google Gemini (Opzionale)")
        gemini_key = st.text_input(
            "API Key",
            type="password",
            help="Solo se vuoi usare modelli Gemini",
            key="gemini_key"
        )

    st.divider()

    # ===== MODEL SELECTION =====
    st.markdown("## 🤖 Modello AI")

    # Crea lista modelli disponibili
    available_models = {}
    if openai_key:
        available_models.update(OPENAI_MODELS)
    if gemini_key:
        available_models.update(GEMINI_MODELS)

    if not available_models:
        st.warning("Inserisci almeno una API key per selezionare un modello")
        model_options = list(OPENAI_MODELS.keys())
    else:
        model_options = list(available_models.keys())

    selected_model = st.selectbox(
        "Seleziona modello",
        options=model_options,
        format_func=lambda x: f"{ALL_MODELS[x]['name']} ({ALL_MODELS[x]['provider'].upper()})",
        help="Modello da usare per ottimizzazione"
    )

    model_info = ALL_MODELS.get(selected_model, {})
    st.caption(f"ℹ️ {model_info.get('description', '')}")

    st.divider()

    # ===== PARAMETRI ANALISI =====
    st.markdown("## ⚙️ Parametri Analisi")

    keyword = st.text_input(
        "🔍 Keyword *",
        placeholder="mutuo partita iva",
        help="Keyword da analizzare (obbligatoria)"
    )

    target_url = st.text_input(
        "🔗 URL Target (opzionale)",
        placeholder="https://example.com/article",
        help="URL della pagina da ottimizzare"
    )

    user_answer = st.text_area(
        "📝 La tua risposta (opzionale)",
        placeholder="Inserisci qui la tua risposta attuale se ne hai già una...",
        help="Se hai già un contenuto, inseriscilo qui",
        height=100
    )

    col_loc, col_lang = st.columns(2)
    with col_loc:
        location = st.selectbox(
            "📍 Location",
            options=list(LOCATION_CODES.keys()),
            index=0
        )

    with col_lang:
        language = st.selectbox(
            "🌐 Language",
            options=list(LANGUAGE_CODES.keys()),
            index=0
        )

    st.divider()

    # ===== PARAMETRI AVANZATI =====
    st.markdown("## 🎛️ Parametri Avanzati")

    max_iterations = st.slider(
        "🔄 Iterazioni ottimizzazione",
        min_value=1,
        max_value=MAX_ITERATIONS,
        value=DEFAULT_ITERATIONS,
        help="Numero di cicli di ottimizzazione"
    )

    max_serp_results = st.slider(
        "📊 Risultati SERP",
        min_value=5,
        max_value=MAX_SERP_RESULTS,
        value=DEFAULT_SERP_RESULTS,
        help="Numero risultati organici da analizzare"
    )

    max_sources = st.slider(
        "🌐 Fonti AIO da scrapare",
        min_value=3,
        max_value=MAX_SOURCES,
        value=5,
        help="Numero massimo fonti AI Overview"
    )

    st.divider()

    # ===== VALIDAZIONE =====
    credentials_valid = all([dataforseo_login, dataforseo_password, openai_key])
    input_valid = bool(keyword)

    if not credentials_valid:
        st.warning("⚠️ Inserisci DataForSEO e OpenAI keys")
    elif not input_valid:
        st.info("📝 Inserisci la keyword da analizzare")
    else:
        st.success("✅ Pronto per l'analisi")

    st.divider()

    # ===== BOTTONE ANALISI =====
    analyze_button = st.button(
        "🚀 Avvia Analisi",
        use_container_width=True,
        disabled=not (credentials_valid and input_valid) or st.session_state.running
    )

    # ===== LOG CONSOLE =====
    st.divider()
    st.markdown("## 📋 Console Log")

    log_container = st.container()
    with log_container:
        st.markdown(
            f'<div class="log-container">{render_logs() or "Nessun log..."}</div>',
            unsafe_allow_html=True
        )

    if st.button("🗑️ Pulisci log", use_container_width=True):
        st.session_state.logs = []
        st.rerun()

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 10])
with col1:
    st.image(MOCA_LOGO_URL, width=60)
with col2:
    st.title("🚀 AI Overview Content Optimizer v2.0")
    st.markdown("**Ottimizza contenuti per Google AI Overview** | by [Moca Interactive](https://mocainteractive.com)")

st.divider()

# ==================== ANALISI ====================
if analyze_button:
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.results = None

    add_log("Inizializzazione workflow...", level="info", agent="orchestrator")

    # Crea stato iniziale
    state = AgentState(
        keyword=keyword,
        target_url=target_url if target_url else None,
        user_answer=user_answer if user_answer else None,
        location=location,
        language=language,
        model_id=selected_model,
        max_iterations=max_iterations,
        max_serp_results=max_serp_results,
        max_sources=max_sources,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key if gemini_key else "",
        dataforseo_login=dataforseo_login,
        dataforseo_password=dataforseo_password
    )

    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔍 Analisi in corso...")
        progress_bar.progress(10)

        # Esegui workflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_state = loop.run_until_complete(run_optimization(state))
        loop.close()

        progress_bar.progress(100)
        status_text.text("✅ Analisi completata!")

        st.session_state.results = result_state
        st.session_state.running = False

    except Exception as e:
        st.error(f"❌ Errore: {str(e)}")
        add_log(f"Errore: {str(e)}", level="error", agent="system")
        st.session_state.running = False

    st.rerun()

# ==================== RISULTATI ====================
if st.session_state.results:
    state = st.session_state.results

    st.markdown("---")
    st.header("📊 RISULTATI ANALISI")

    # ========== METRICHE OVERVIEW ==========
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state.best_score:.1%}</div>
            <div class="metric-label">Score Finale</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        initial_score = state.initial_ranking[0]["score"] if state.initial_ranking else 0
        improvement = ((state.best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{improvement:+.1f}%</div>
            <div class="metric-label">Miglioramento</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(state.iterations)}</div>
            <div class="metric-label">Iterazioni</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(state.ai_overview_sources)}</div>
            <div class="metric-label">Fonti AIO</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== AI OVERVIEW ==========
    if state.ai_overview_text:
        st.markdown("---")
        st.subheader("🤖 AI Overview di Google")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(state.ai_overview_text)
        with col2:
            st.metric("Caratteri", len(state.ai_overview_text))
            st.metric("Fonti", len(state.ai_overview_sources))

    # ========== RANKING INIZIALE ==========
    if state.initial_ranking:
        st.markdown("---")
        st.subheader("📊 Ranking Iniziale")

        ranking_df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Tipo": r["type"],
                "Fonte": r["label"],
                "Score": f"{r['score']:.4f}"
            }
            for r in state.initial_ranking[:10]
        ])
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    # ========== ITERAZIONI ==========
    if state.iterations:
        st.markdown("---")
        st.subheader("🔄 Cicli di Ottimizzazione")

        for iteration in state.iterations:
            with st.expander(f"📝 Iterazione {iteration['iteration']} - Score: {iteration['score']:.4f} ({iteration['improvement']:+.2f}%)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Ragionamento:**")
                    st.info(iteration["reasoning"][:500] + "..." if len(iteration["reasoning"]) > 500 else iteration["reasoning"])

                    st.markdown("**Risposta Ottimizzata:**")
                    st.success(iteration["answer"])

                with col2:
                    st.markdown("**Ranking Iterazione:**")
                    if iteration.get("ranking"):
                        for r in iteration["ranking"][:5]:
                            st.write(f"#{r['rank']} {r['label']}: {r['score']:.4f}")

    # ========== RISPOSTA FINALE ==========
    if state.best_answer:
        st.markdown("---")
        st.subheader("✨ Risposta Ottimizzata Finale")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(state.best_answer)
        with col2:
            st.metric("Score Finale", f"{state.best_score:.4f}")
            st.metric("Parole", len(state.best_answer.split()))

    # ========== RANKING FINALE ==========
    if state.current_ranking:
        st.markdown("---")
        st.subheader("🏆 Ranking Finale")

        final_ranking_df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Tipo": r["type"],
                "Fonte": r["label"],
                "Score": f"{r['score']:.4f}"
            }
            for r in state.current_ranking[:10]
        ])
        st.dataframe(final_ranking_df, use_container_width=True, hide_index=True)

    # ========== ANALISI STRATEGICA ==========
    if state.strategic_analysis:
        st.markdown("---")
        st.subheader("📋 Analisi Strategica")

        with st.expander("Visualizza Analisi Completa"):
            st.markdown("**Ragionamento:**")
            st.info(state.strategic_analysis.get("reasoning", ""))
            st.markdown("**Analisi:**")
            st.write(state.strategic_analysis.get("analysis", ""))

    # ========== PIANO CONTENUTO ==========
    if state.content_plan:
        st.markdown("---")
        st.subheader("📝 Piano Contenuto")

        with st.expander("Visualizza Piano Completo"):
            st.markdown("**Ragionamento:**")
            st.info(state.content_plan.get("reasoning", ""))
            st.markdown("**Piano:**")
            st.write(state.content_plan.get("plan", ""))

    # ========== EXPORT ==========
    st.markdown("---")
    st.subheader("📥 Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON Export
        export_data = {
            "keyword": state.keyword,
            "best_answer": state.best_answer,
            "best_score": state.best_score,
            "iterations": state.iterations,
            "initial_ranking": state.initial_ranking,
            "final_ranking": state.current_ranking,
            "ai_overview_text": state.ai_overview_text,
            "strategic_analysis": state.strategic_analysis,
            "content_plan": state.content_plan,
            "generated_at": datetime.now().isoformat()
        }

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"ai_overview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        # CSV Summary
        summary_data = {
            "Keyword": [state.keyword],
            "Best Score": [state.best_score],
            "Iterations": [len(state.iterations)],
            "AIO Sources": [len(state.ai_overview_sources)],
            "Timestamp": [datetime.now().isoformat()]
        }
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col3:
        # PDF Export
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Overview Optimization Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Keyword: {state.keyword}", ln=True)
            pdf.cell(0, 10, f"Score Finale: {state.best_score:.4f}", ln=True)
            pdf.cell(0, 10, f"Iterazioni: {len(state.iterations)}", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Risposta Ottimizzata:", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, state.best_answer[:1000] if state.best_answer else "N/A")

            pdf_output = pdf.output(dest='S').encode('latin-1')

            st.download_button(
                label="📑 Download PDF",
                data=pdf_output,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning("PDF export non disponibile")

# ==================== ISTRUZIONI INIZIALI ====================
if not st.session_state.results and not st.session_state.running:
    st.info("👈 **Per iniziare**: Configura le API keys e inserisci la keyword nella sidebar")

    st.markdown("""
    ## 🚀 Come Funziona

    Questo strumento usa un **sistema multi-agente** per ottimizzare i tuoi contenuti per Google AI Overview.

    ### Workflow:

    1. 🔍 **SERP Analysis** - Recupera AI Overview e dati SERP
    2. 🕷️ **Content Scraping** - Estrae contenuti dalle fonti top
    3. 📊 **Ranking Iniziale** - Calcola posizionamento iniziale
    4. 🔄 **Ciclo Ottimizzazione** - Migliora iterativamente con reasoning
    5. 📋 **Analisi Strategica** - Intento, competitor, opportunità
    6. 📝 **Piano Contenuto** - Struttura articolo H1/H2/H3

    ### Modelli Supportati:
    - **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, o1-preview, o1-mini
    - **Gemini**: 1.5 Pro, 1.5 Flash, 2.0 Flash

    ### Caratteristiche:
    - ✅ Ottimizzazione iterativa con ragionamento
    - ✅ Ranking comparativo ad ogni iterazione
    - ✅ Analisi strategica competitor
    - ✅ Piano contenuto strutturato
    - ✅ Export JSON/CSV/PDF
    """)

# ==================== FOOTER ====================
st.divider()
st.markdown(f"""
<div style='text-align: center; color: {MOCA_COLORS['gray']}; padding: 20px;'>
    <p>Sviluppato da <a href='https://mocainteractive.com' target='_blank' style='color: {MOCA_COLORS['primary']}; text-decoration: none;'><strong>Moca Interactive</strong></a></p>
    <p style='font-size: 0.9em;'>© 2025 Moca Interactive. Tutti i diritti riservati.</p>
    <p style='font-size: 0.8em;'>v2.0 - Multi-Agent System</p>
</div>
""", unsafe_allow_html=True)
