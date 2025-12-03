"""
AI Overview Content Optimizer v2.2 - Multi-Agent System
Phase-based display with full logging
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
import time

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
    page_title="AI Overview Optimizer v2.2 | Moca",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'phase_logs' not in st.session_state:
    st.session_state.phase_logs = {
        "serp": [],
        "scraping": [],
        "ranking": [],
        "optimization": [],
        "strategic": [],
        "content_plan": []
    }
if 'results' not in st.session_state:
    st.session_state.results = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = ""

# ==================== CUSTOM CSS ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Figtree', sans-serif;
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

    .log-mini {{
        background-color: #1e1e2e;
        color: #89b4fa;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.7rem;
        padding: 8px 12px;
        border-radius: 6px;
        max-height: 200px;
        overflow-y: auto;
        margin: 5px 0;
    }}

    .phase-box {{
        background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
        border: 1px solid #e5e7eb;
        border-left: 4px solid {MOCA_COLORS['primary']};
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }}

    .phase-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {MOCA_COLORS['primary']};
        margin-bottom: 10px;
    }}

    .score-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }}

    .score-high {{ background-color: #d4edda; color: #155724; }}
    .score-medium {{ background-color: #fff3cd; color: #856404; }}
    .score-low {{ background-color: #f8d7da; color: #721c24; }}

    .reasoning-box {{
        background: #f0f7ff;
        border: 1px solid #bdd7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }}

    .answer-box {{
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }}

    .competitor-box {{
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
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

    .ranking-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }}

    .ranking-table th {{
        background-color: {MOCA_COLORS['primary']};
        color: white;
        padding: 10px;
        text-align: left;
    }}

    .ranking-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #eee;
    }}

    .ranking-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def get_score_class(score: float) -> str:
    if score >= 0.85:
        return "score-high"
    elif score >= 0.70:
        return "score-medium"
    return "score-low"


def format_score_badge(score: float) -> str:
    score_class = get_score_class(score)
    return f'<span class="score-badge {score_class}">{score:.4f}</span>'


def render_phase_logs(phase: str) -> str:
    """Render logs per una fase specifica"""
    logs = st.session_state.phase_logs.get(phase, [])
    if not logs:
        return ""

    html = []
    for log in logs[-30:]:
        color = {"info": "#89b4fa", "success": "#a6e3a1", "warning": "#f9e2af", "error": "#f38ba8"}.get(log['level'], '#cdd6f4')
        html.append(f'<div style="color:{color}">• {log["message"]}</div>')
    return "".join(html)


# ==================== LOG CALLBACK ====================
class PhaseLogCallback:
    """Callback che organizza i log per fase"""

    def __init__(self):
        self.current_phase = "serp"

    def __call__(self, log_entry: Dict):
        msg = log_entry.get("message", "")

        # Determina la fase dal messaggio
        if "FASE 1" in msg or "SERP" in msg.upper():
            self.current_phase = "serp"
        elif "FASE 2" in msg or "Scraping" in msg:
            self.current_phase = "scraping"
        elif "FASE 3" in msg or "Ranking Iniziale" in msg:
            self.current_phase = "ranking"
        elif "FASE 4" in msg or "Ottimizzazione" in msg or "ITERAZIONE" in msg:
            self.current_phase = "optimization"
        elif "FASE 5" in msg or "Strategica" in msg:
            self.current_phase = "strategic"
        elif "FASE 6" in msg or "Piano" in msg:
            self.current_phase = "content_plan"

        # Aggiungi log alla fase corrente
        st.session_state.phase_logs[self.current_phase].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": log_entry.get("level", "info"),
            "message": msg
        })

        # Aggiungi anche al log globale
        st.session_state.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": log_entry.get("agent", "agent"),
            "level": log_entry.get("level", "info"),
            "message": msg
        })

        st.session_state.current_phase = self.current_phase


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image(MOCA_LOGO_URL, width=80)
    st.markdown("# 🚀 AI Overview Optimizer")
    st.markdown("**v2.2 - Phase Display**")

    st.divider()

    # API KEYS
    with st.expander("🔑 API Keys", expanded=True):
        dataforseo_login = st.text_input("DataForSEO Login", key="dataforseo_login")
        dataforseo_password = st.text_input("DataForSEO Password", type="password", key="dataforseo_password")
        st.markdown("---")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        st.markdown("---")
        gemini_key = st.text_input("Gemini API Key (opzionale)", type="password", key="gemini_key")

    st.divider()

    # MODEL SELECTION
    st.markdown("## 🤖 Modello")
    available_models = {}
    if openai_key:
        available_models.update(OPENAI_MODELS)
    if gemini_key:
        available_models.update(GEMINI_MODELS)

    model_options = list(available_models.keys()) if available_models else list(OPENAI_MODELS.keys())
    selected_model = st.selectbox(
        "Modello",
        options=model_options,
        format_func=lambda x: f"{ALL_MODELS[x]['name']}"
    )

    st.divider()

    # PARAMETRI
    st.markdown("## ⚙️ Parametri")
    keyword = st.text_input("🔍 Keyword *", placeholder="mutuo partita iva")
    target_url = st.text_input("🔗 URL Target (opz.)", placeholder="https://...")
    user_answer = st.text_area("📝 Tua risposta (opz.)", height=80)

    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("📍 Location", list(LOCATION_CODES.keys()))
    with col2:
        language = st.selectbox("🌐 Lingua", list(LANGUAGE_CODES.keys()))

    st.divider()

    # AVANZATI
    max_iterations = st.slider("🔄 Iterazioni", 1, MAX_ITERATIONS, DEFAULT_ITERATIONS)
    max_serp_results = st.slider("📊 Risultati SERP", 5, MAX_SERP_RESULTS, DEFAULT_SERP_RESULTS)
    max_sources = st.slider("🌐 Fonti AIO", 3, MAX_SOURCES, 5)

    st.divider()

    # VALIDAZIONE
    credentials_valid = all([dataforseo_login, dataforseo_password, openai_key])
    input_valid = bool(keyword)

    if credentials_valid and input_valid:
        st.success("✅ Pronto")
    else:
        st.warning("⚠️ Completa i campi richiesti")

    analyze_button = st.button(
        "🚀 Avvia Analisi",
        use_container_width=True,
        disabled=not (credentials_valid and input_valid) or st.session_state.running
    )

# ==================== MAIN CONTENT ====================
st.title("🚀 AI Overview Content Optimizer v2.2")
st.markdown("**Ottimizza contenuti per Google AI Overview** | by [Moca Interactive](https://mocainteractive.com)")
st.divider()

# ==================== ESECUZIONE ANALISI ====================
if analyze_button:
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.phase_logs = {k: [] for k in st.session_state.phase_logs}
    st.session_state.results = None

    # Crea stato
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
    progress = st.progress(0, text="Inizializzazione...")
    status = st.empty()

    try:
        log_callback = PhaseLogCallback()
        orchestrator = OrchestratorAgent(log_callback=log_callback)

        progress.progress(10, text="📡 Recupero SERP e AI Overview...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_state = loop.run_until_complete(orchestrator.run(state))
        loop.close()

        progress.progress(100, text="✅ Completato!")
        st.session_state.results = result_state
        st.session_state.running = False

        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Errore: {e}")
        st.session_state.running = False
        import traceback
        st.code(traceback.format_exc())

# ==================== RISULTATI ====================
if st.session_state.results:
    state = st.session_state.results

    # Calcola metriche
    initial_user_score = 0
    if state.initial_ranking:
        user_item = next((r for r in state.initial_ranking if r["type"] == "user_answer"), None)
        if user_item:
            initial_user_score = user_item["score"]

    improvement = ((state.best_score - initial_user_score) / initial_user_score * 100) if initial_user_score > 0 else 0

    # ==================== METRICHE ====================
    st.header("📊 Risultati")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{state.best_score:.1%}</div>
            <div class="metric-label">Score Finale</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        color = "#10b981" if improvement >= 0 else "#ef4444"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{improvement:+.1f}%</div>
            <div class="metric-label">Miglioramento</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(state.iterations)}</div>
            <div class="metric-label">Iterazioni</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(state.competitor_contents)}</div>
            <div class="metric-label">Competitor</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ==================== FASE 1: AI OVERVIEW ====================
    st.markdown("### 🎯 FASE 1: AI Overview di Google")

    with st.expander("📋 Log Fase 1 - SERP", expanded=False):
        st.markdown(f'<div class="log-mini">{render_phase_logs("serp")}</div>', unsafe_allow_html=True)

    if state.ai_overview_text:
        st.markdown(f"""<div class="phase-box">
            <div class="phase-title">Testo AI Overview (riferimento)</div>
            <p>{state.ai_overview_text}</p>
            <small>📊 {len(state.ai_overview_text)} caratteri | {len(state.ai_overview_sources)} fonti</small>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("AI Overview non disponibile per questa keyword")

    st.divider()

    # ==================== FASE 2: COMPETITOR ====================
    st.markdown("### 🏢 FASE 2: Risposte Competitor")

    with st.expander("📋 Log Fase 2 - Scraping", expanded=False):
        st.markdown(f'<div class="log-mini">{render_phase_logs("scraping")}</div>', unsafe_allow_html=True)

    if state.competitor_contents:
        for i, comp in enumerate(state.competitor_contents):
            # Trova score competitor
            comp_score = None
            for r in state.initial_ranking:
                if comp['domain'] in r.get('label', ''):
                    comp_score = r['score']
                    break

            with st.expander(f"**{i+1}. {comp['domain']}** {format_score_badge(comp_score) if comp_score else ''}", expanded=(i < 2)):
                st.markdown(f"**URL:** [{comp['url'][:50]}...]({comp['url']})")
                st.markdown(f"**Parole:** {comp['word_count']}")
                response = comp.get('response_preview', comp['content'][:500])
                st.markdown(f"""<div class="competitor-box">{response}</div>""", unsafe_allow_html=True)
    else:
        st.info("Nessun competitor trovato")

    st.divider()

    # ==================== FASE 3: RANKING INIZIALE ====================
    st.markdown("### 📊 FASE 3: Ranking Iniziale")

    with st.expander("📋 Log Fase 3 - Ranking", expanded=False):
        st.markdown(f'<div class="log-mini">{render_phase_logs("ranking")}</div>', unsafe_allow_html=True)

    if state.initial_ranking:
        # Tabella ranking
        html = '<table class="ranking-table"><thead><tr><th>#</th><th>Fonte</th><th>Tipo</th><th>Score</th></tr></thead><tbody>'
        for r in state.initial_ranking:
            if r.get("is_reference"):
                continue
            emoji = "📝" if r["type"] == "user_answer" else "🏢"
            html += f'<tr><td>{r["rank"]}</td><td>{emoji} {r["label"]}</td><td>{r["type"]}</td><td>{format_score_badge(r["score"])}</td></tr>'
        html += '</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)

        if initial_user_score > 0:
            st.info(f"📍 **La tua posizione iniziale:** Score {initial_user_score:.4f}")
    else:
        st.warning("Ranking non disponibile (serve una risposta iniziale)")

    st.divider()

    # ==================== FASE 4: ITERAZIONI ====================
    st.markdown("### 🔄 FASE 4: Ciclo di Ottimizzazione")

    with st.expander("📋 Log Fase 4 - Ottimizzazione", expanded=False):
        st.markdown(f'<div class="log-mini">{render_phase_logs("optimization")}</div>', unsafe_allow_html=True)

    if state.iterations:
        # Tabella evoluzione score
        st.markdown("#### 📈 Evoluzione Score")

        evolution_data = []
        for it in state.iterations:
            evolution_data.append({
                "Iterazione": it["iteration"],
                "Score": f"{it['score']:.4f}",
                "Variazione": f"{it.get('improvement', 0):+.2f}%"
            })
        st.dataframe(pd.DataFrame(evolution_data), use_container_width=True, hide_index=True)

        # Dettaglio iterazioni
        st.markdown("#### 📝 Dettaglio Iterazioni")

        for it in state.iterations:
            iter_num = it['iteration']
            score = it['score']
            imp = it.get('improvement', 0)
            color = "#10b981" if imp >= 0 else "#ef4444"

            st.markdown(f"##### Iterazione {iter_num} - Score: {score:.4f} <span style='color:{color}'>({imp:+.2f}%)</span>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**💭 Ragionamento:**")
                reasoning = it.get("reasoning", "Nessun ragionamento disponibile")
                st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("**📝 Risposta Generata:**")
                answer = it.get("answer", "Nessuna risposta")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # Ranking dopo iterazione
            if it.get("ranking"):
                with st.expander(f"📊 Ranking dopo iterazione {iter_num}"):
                    mini_df = pd.DataFrame([
                        {"#": r["rank"], "Fonte": r["label"], "Score": f"{r['score']:.4f}"}
                        for r in it["ranking"][:6]
                    ])
                    st.dataframe(mini_df, use_container_width=True, hide_index=True)

            st.markdown("---")
    else:
        st.info("Nessuna iterazione eseguita")

    # ==================== RISPOSTA FINALE ====================
    st.markdown("### ✨ RISPOSTA OTTIMIZZATA FINALE")

    if state.best_answer:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="answer-box" style="font-size: 1.1rem;">{state.best_answer}</div>', unsafe_allow_html=True)
        with col2:
            st.metric("Score Finale", f"{state.best_score:.4f}")
            st.metric("Parole", len(state.best_answer.split()))
            if initial_user_score > 0:
                st.metric("vs Iniziale", f"{improvement:+.1f}%")
    else:
        st.warning("Nessuna risposta ottimizzata generata")

    st.divider()

    # ==================== FASE 5 & 6: ANALISI ====================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 FASE 5: Analisi Strategica")
        with st.expander("📋 Log Fase 5", expanded=False):
            st.markdown(f'<div class="log-mini">{render_phase_logs("strategic")}</div>', unsafe_allow_html=True)

        if state.strategic_analysis:
            st.markdown("**Ragionamento:**")
            st.markdown(f'<div class="reasoning-box">{state.strategic_analysis.get("reasoning", "")}</div>', unsafe_allow_html=True)
            st.markdown("**Analisi:**")
            st.write(state.strategic_analysis.get("analysis", ""))
        else:
            st.info("Analisi non disponibile")

    with col2:
        st.markdown("### 📝 FASE 6: Piano Contenuto")
        with st.expander("📋 Log Fase 6", expanded=False):
            st.markdown(f'<div class="log-mini">{render_phase_logs("content_plan")}</div>', unsafe_allow_html=True)

        if state.content_plan:
            st.markdown("**Ragionamento:**")
            st.markdown(f'<div class="reasoning-box">{state.content_plan.get("reasoning", "")}</div>', unsafe_allow_html=True)
            st.markdown("**Piano:**")
            st.write(state.content_plan.get("plan", ""))
        else:
            st.info("Piano non disponibile")

    st.divider()

    # ==================== EXPORT ====================
    st.markdown("### 📥 Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        export_data = {
            "keyword": state.keyword,
            "best_answer": state.best_answer,
            "best_score": state.best_score,
            "initial_score": initial_user_score,
            "improvement": improvement,
            "iterations": state.iterations,
            "ai_overview_text": state.ai_overview_text,
            "competitor_responses": [
                {"domain": c["domain"], "response": c.get("response_preview", "")}
                for c in state.competitor_contents
            ],
            "generated_at": datetime.now().isoformat()
        }
        st.download_button(
            "📄 JSON",
            json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

    with col2:
        csv_data = pd.DataFrame([{
            "Keyword": state.keyword,
            "Score Iniziale": initial_user_score,
            "Score Finale": state.best_score,
            "Miglioramento": f"{improvement:.2f}%",
            "Iterazioni": len(state.iterations)
        }]).to_csv(index=False)
        st.download_button("📊 CSV", csv_data, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

    with col3:
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Overview Optimization Report", ln=True, align="C")
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"Keyword: {state.keyword}", ln=True)
            pdf.cell(0, 8, f"Score: {initial_user_score:.4f} -> {state.best_score:.4f} ({improvement:+.1f}%)", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Risposta Ottimizzata:", ln=True)
            pdf.set_font("Arial", "", 10)
            safe_text = (state.best_answer or "N/A").encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_text[:3000])
            st.download_button("📑 PDF", pdf.output(dest='S').encode('latin-1', 'replace'),
                               f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")
        except:
            st.warning("PDF non disponibile")

# ==================== ISTRUZIONI ====================
if not st.session_state.results and not st.session_state.running:
    st.info("👈 Configura le API keys e inserisci la keyword per iniziare")

    st.markdown("""
    ## Come Funziona v2.2

    **Workflow in 6 Fasi:**
    1. 🎯 **SERP & AI Overview** - Recupera il riferimento da Google
    2. 🏢 **Competitor Scraping** - Estrae le risposte dei competitor
    3. 📊 **Ranking Iniziale** - Calcola la tua posizione
    4. 🔄 **Ottimizzazione Iterativa** - Migliora progressivamente (ogni iterazione parte dalla precedente)
    5. 📋 **Analisi Strategica** - Gap e opportunità
    6. 📝 **Piano Contenuto** - Struttura H1/H2/H3

    **Novità v2.2:**
    - ✅ Risultati divisi per fase
    - ✅ Log separati per ogni fase
    - ✅ Ragionamento completo (non troncato)
    - ✅ Risposte pulite e originali
    - ✅ Tabella evoluzione score
    """)

# ==================== FOOTER ====================
st.divider()
st.markdown(f"""<div style='text-align: center; color: {MOCA_COLORS['gray']}; padding: 10px;'>
    <a href='https://mocainteractive.com' target='_blank' style='color: {MOCA_COLORS['primary']};'>Moca Interactive</a> | v2.2
</div>""", unsafe_allow_html=True)
