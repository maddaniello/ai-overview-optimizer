"""
AI Overview Content Optimizer v2.3
Real-time logging + Dark mode compatible
Developed by Moca Interactive
"""
import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict, List
import time

sys.path.insert(0, str(Path(__file__).parent))

from agents.base_agent import AgentState
from agents.orchestrator import OrchestratorAgent
from config import (
    MOCA_COLORS, MOCA_LOGO_URL, ALL_MODELS, OPENAI_MODELS, GEMINI_MODELS,
    LOCATION_CODES, LANGUAGE_CODES, DEFAULT_ITERATIONS, MAX_ITERATIONS,
    DEFAULT_SERP_RESULTS, MAX_SERP_RESULTS, MAX_SOURCES
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Overview Optimizer | Moca",
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

# ==================== MINIMAL CSS (dark mode safe) ====================
st.markdown("""
<style>
    .log-box {
        background-color: #0d1117;
        color: #58a6ff;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.75rem;
        padding: 12px;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #30363d;
    }
    .log-info { color: #58a6ff; }
    .log-success { color: #3fb950; }
    .log-warning { color: #d29922; }
    .log-error { color: #f85149; }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def render_logs_html(logs: List[Dict]) -> str:
    """Render logs as HTML"""
    if not logs:
        return '<span style="color:#8b949e">In attesa...</span>'

    html = []
    for log in logs[-100:]:
        colors = {
            "info": "#58a6ff",
            "success": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149"
        }
        color = colors.get(log.get('level', 'info'), '#8b949e')
        msg = log.get('message', '').replace('<', '&lt;').replace('>', '&gt;')
        html.append(f'<div style="color:{color}; margin:2px 0;">• {msg}</div>')

    return "".join(html)


def render_ranking_table(ranking: List[Dict], title: str = "Ranking"):
    """Render ranking table with Streamlit"""
    if not ranking:
        return

    st.markdown(f"**{title}**")

    data = []
    for r in ranking:
        if r.get("is_reference"):
            continue
        emoji = "📝" if r["type"] == "user_answer" else ("✨" if r["type"] == "optimized" else "🏢")
        data.append({
            "#": r["rank"],
            "Tipo": emoji,
            "Fonte": r["label"],
            "Score": f"{r['score']:.4f}"
        })

    if data:
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image(MOCA_LOGO_URL, width=80)
    st.markdown("## AI Overview Optimizer")

    st.divider()

    # API KEYS
    with st.expander("🔑 API Keys", expanded=True):
        dataforseo_login = st.text_input("DataForSEO Login", key="dfs_login")
        dataforseo_password = st.text_input("DataForSEO Password", type="password", key="dfs_pass")
        st.markdown("---")
        openai_key = st.text_input("OpenAI API Key", type="password", key="oai_key")
        st.markdown("---")
        gemini_key = st.text_input("Gemini Key (opz.)", type="password", key="gem_key")

    st.divider()

    # MODEL
    st.markdown("### Modello")
    available_models = {}
    if openai_key:
        available_models.update(OPENAI_MODELS)
    if gemini_key:
        available_models.update(GEMINI_MODELS)

    model_options = list(available_models.keys()) if available_models else list(OPENAI_MODELS.keys())
    selected_model = st.selectbox("Modello", options=model_options,
                                   format_func=lambda x: ALL_MODELS[x]['name'])

    st.divider()

    # PARAMS
    st.markdown("### Parametri")
    keyword = st.text_input("🔍 Keyword *", placeholder="mutuo partita iva")
    target_url = st.text_input("🔗 URL Target (opz.)")
    user_answer = st.text_area("📝 Tua risposta (opz.)", height=80)

    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Location", list(LOCATION_CODES.keys()))
    with col2:
        language = st.selectbox("Lingua", list(LANGUAGE_CODES.keys()))

    st.divider()

    max_iterations = st.slider("Iterazioni", 1, MAX_ITERATIONS, DEFAULT_ITERATIONS)
    max_sources = st.slider("Fonti AIO", 3, MAX_SOURCES, 5)

    st.divider()

    # VALIDATION
    ready = all([dataforseo_login, dataforseo_password, openai_key, keyword])

    if ready:
        st.success("✅ Pronto")
    else:
        st.warning("⚠️ Completa i campi")

    analyze_btn = st.button("🚀 Avvia Analisi", use_container_width=True,
                            disabled=not ready or st.session_state.running)

# ==================== MAIN ====================
st.title("🚀 AI Overview Optimizer")
st.caption("by [Moca Interactive](https://mocainteractive.com)")
st.divider()

# Placeholders for real-time updates
log_placeholder = st.empty()
content_placeholder = st.container()

# ==================== EXECUTION ====================
if analyze_btn:
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.results = None

    # Log callback that updates UI in real-time
    def log_callback(entry: Dict):
        st.session_state.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": entry.get("level", "info"),
            "message": entry.get("message", "")
        })
        # Update log display
        log_placeholder.markdown(
            f'<div class="log-box">{render_logs_html(st.session_state.logs)}</div>',
            unsafe_allow_html=True
        )

    # Initial log display
    log_placeholder.markdown(
        f'<div class="log-box">{render_logs_html(st.session_state.logs)}</div>',
        unsafe_allow_html=True
    )

    # Create state
    state = AgentState(
        keyword=keyword,
        target_url=target_url if target_url else None,
        user_answer=user_answer if user_answer else None,
        location=location,
        language=language,
        model_id=selected_model,
        max_iterations=max_iterations,
        max_serp_results=DEFAULT_SERP_RESULTS,
        max_sources=max_sources,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key if gemini_key else "",
        dataforseo_login=dataforseo_login,
        dataforseo_password=dataforseo_password
    )

    try:
        orchestrator = OrchestratorAgent(log_callback=log_callback)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_state = loop.run_until_complete(orchestrator.run(state))
        loop.close()

        st.session_state.results = result_state
        st.session_state.running = False

        st.rerun()

    except Exception as e:
        st.error(f"Errore: {e}")
        st.session_state.running = False
        import traceback
        st.code(traceback.format_exc())

# ==================== RESULTS ====================
if st.session_state.results:
    state = st.session_state.results

    # Show final log
    with st.expander("📋 Log Esecuzione", expanded=False):
        st.markdown(
            f'<div class="log-box">{render_logs_html(st.session_state.logs)}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ===== AI OVERVIEW =====
    st.subheader("🎯 AI Overview di Google")
    if state.ai_overview_text:
        st.info(state.ai_overview_text)
        st.caption(f"{len(state.ai_overview_text)} caratteri | {len(state.ai_overview_sources)} fonti")
    else:
        st.warning("AI Overview non disponibile")

    st.divider()

    # ===== COMPETITOR =====
    st.subheader("🏢 Risposte Competitor")
    if state.competitor_contents:
        for i, comp in enumerate(state.competitor_contents):
            with st.expander(f"{i+1}. {comp['domain']}", expanded=(i < 2)):
                st.caption(f"URL: {comp['url']}")
                st.write(comp.get('response_preview', comp['content'][:500]))
    else:
        st.info("Nessun competitor")

    st.divider()

    # ===== RANKING INIZIALE =====
    st.subheader("📊 Ranking Iniziale")
    render_ranking_table(state.initial_ranking, "Posizionamento vs AI Overview")

    st.divider()

    # ===== ITERAZIONI =====
    st.subheader("🔄 Iterazioni di Ottimizzazione")

    if state.iterations:
        for it in state.iterations:
            iter_num = it['iteration']
            score = it['score']

            st.markdown(f"### Iterazione {iter_num} - Score: {score:.4f}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Ragionamento:**")
                st.info(it.get("reasoning", "N/A"))

            with col2:
                st.markdown("**Risposta:**")
                st.success(it.get("answer", "N/A"))

            # Ranking dopo questa iterazione
            if it.get("ranking"):
                render_ranking_table(it["ranking"], f"Ranking dopo iterazione {iter_num}")

            st.divider()
    else:
        st.info("Nessuna iterazione")

    # ===== RISPOSTA FINALE =====
    st.subheader("✨ Risposta Ottimizzata Finale")
    if state.best_answer:
        st.success(state.best_answer)
        st.caption(f"Score: {state.best_score:.4f} | Parole: {len(state.best_answer.split())}")
    else:
        st.warning("Nessuna risposta generata")

    st.divider()

    # ===== ANALISI STRATEGICA =====
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Analisi Strategica")
        if state.strategic_analysis:
            st.info(state.strategic_analysis.get("reasoning", ""))
            st.write(state.strategic_analysis.get("analysis", ""))
        else:
            st.info("Non disponibile")

    with col2:
        st.subheader("📝 Piano Contenuto")
        if state.content_plan:
            st.info(state.content_plan.get("reasoning", ""))
            st.write(state.content_plan.get("plan", ""))
        else:
            st.info("Non disponibile")

    st.divider()

    # ===== EXPORT =====
    st.subheader("📥 Export")

    col1, col2 = st.columns(2)

    with col1:
        export_data = {
            "keyword": state.keyword,
            "best_answer": state.best_answer,
            "best_score": state.best_score,
            "iterations": state.iterations,
            "ai_overview_text": state.ai_overview_text,
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
            "Score Finale": state.best_score,
            "Iterazioni": len(state.iterations)
        }]).to_csv(index=False)
        st.download_button("📊 CSV", csv_data, "summary.csv", "text/csv")

# ==================== INSTRUCTIONS ====================
if not st.session_state.results and not st.session_state.running:
    st.info("👈 Configura API keys e keyword per iniziare")

    st.markdown("""
    ### Workflow
    1. 🎯 Recupera AI Overview da Google
    2. 🏢 Scraping risposte competitor
    3. 📊 Ranking iniziale
    4. 🔄 Ottimizzazione iterativa
    5. 📋 Analisi strategica
    6. 📝 Piano contenuto
    """)

# ==================== FOOTER ====================
st.divider()
st.caption("Moca Interactive | AI Overview Optimizer v2.3")
