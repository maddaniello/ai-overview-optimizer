"""
AI Overview Content Optimizer v2.1 - Multi-Agent System
Real-time logging + Ranking Evolution
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
    page_title="AI Overview Optimizer v2.1 | Moca",
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
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = ""

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
        font-size: 0.72rem;
        padding: 12px;
        border-radius: 8px;
        max-height: 500px;
        overflow-y: auto;
        line-height: 1.4;
    }}

    .log-info {{ color: #3b82f6; }}
    .log-success {{ color: #10b981; }}
    .log-warning {{ color: #f59e0b; }}
    .log-error {{ color: #ef4444; }}

    .phase-indicator {{
        background: linear-gradient(135deg, {MOCA_COLORS['primary']} 0%, #ff6b6b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
    }}

    .ranking-evolution-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }}

    .ranking-evolution-table th {{
        background-color: {MOCA_COLORS['primary']};
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }}

    .ranking-evolution-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
    }}

    .ranking-evolution-table tr:hover {{
        background-color: {MOCA_COLORS['secondary']};
    }}

    .score-badge {{
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }}

    .score-high {{ background-color: #d4edda; color: #155724; }}
    .score-medium {{ background-color: #fff3cd; color: #856404; }}
    .score-low {{ background-color: #f8d7da; color: #721c24; }}

    .competitor-card {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}

    .competitor-domain {{
        font-weight: 700;
        color: {MOCA_COLORS['primary']};
        font-size: 1.1em;
    }}

    .competitor-response {{
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
        line-height: 1.5;
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

    .iteration-card {{
        background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
        border-left: 4px solid {MOCA_COLORS['primary']};
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
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
    return log_entry


def render_logs_html():
    """Renderizza i log in formato HTML"""
    if not st.session_state.logs:
        return '<span style="color:#666">In attesa di avvio...</span>'

    html_lines = []
    for log in st.session_state.logs[-100:]:  # Ultimi 100 log
        level_colors = {
            "info": "#3b82f6",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444"
        }
        color = level_colors.get(log['level'], '#666')

        # Emoji per tipo
        emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(log['level'], "•")

        html_lines.append(
            f'<div style="margin: 2px 0;">'
            f'<span style="color:#555">[{log["timestamp"]}]</span> '
            f'<span style="color:{color}; font-weight:600;">[{log["agent"].upper()}]</span> '
            f'{emoji} {log["message"]}'
            f'</div>'
        )

    return "".join(html_lines)


def get_score_class(score: float) -> str:
    """Restituisce classe CSS per score"""
    if score >= 0.85:
        return "score-high"
    elif score >= 0.70:
        return "score-medium"
    return "score-low"


def create_ranking_evolution_table(state: AgentState) -> str:
    """Crea tabella HTML dell'evoluzione del ranking"""
    if not state.initial_ranking:
        return "<p>Nessun dato di ranking disponibile</p>"

    # Intestazioni: Fonte | Iniziale | Iter 1 | Iter 2 | ... | Finale
    num_iterations = len(state.iterations)
    headers = ["Fonte", "Tipo", "Score Iniziale"]
    for i in range(1, num_iterations + 1):
        headers.append(f"Iter. {i}")
    if num_iterations > 0:
        headers.append("Variazione")

    # Costruisci righe
    rows_data = []

    # Prima aggiungi tutti i contenuti dal ranking iniziale
    for item in state.initial_ranking:
        if item.get("is_reference"):
            continue  # Skip AI Overview reference

        row = {
            "label": item["label"],
            "type": item["type"],
            "initial_score": item["score"],
            "iteration_scores": [],
            "url": item.get("url", "")
        }

        # Trova score in ogni iterazione
        for iteration in state.iterations:
            iter_ranking = iteration.get("ranking", [])
            matching = next((r for r in iter_ranking if r["label"] == item["label"]), None)
            if matching:
                row["iteration_scores"].append(matching["score"])
            else:
                row["iteration_scores"].append(None)

        rows_data.append(row)

    # Aggiungi le risposte ottimizzate
    for i, iteration in enumerate(state.iterations):
        iter_ranking = iteration.get("ranking", [])
        optimized = next((r for r in iter_ranking if r["type"] == "optimized"), None)
        if optimized:
            # Controlla se già esiste
            existing = next((r for r in rows_data if f"Ottimizzata" in r["label"]), None)
            if not existing:
                row = {
                    "label": f"Risposta Ottimizzata",
                    "type": "optimized",
                    "initial_score": None,
                    "iteration_scores": [None] * i + [optimized["score"]],
                    "url": state.target_url
                }
                rows_data.append(row)
            else:
                # Aggiorna con nuovo score
                while len(existing["iteration_scores"]) < i:
                    existing["iteration_scores"].append(None)
                existing["iteration_scores"].append(optimized["score"])

    # Costruisci HTML
    html = '<table class="ranking-evolution-table">'
    html += '<thead><tr>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'

    for row in rows_data:
        type_emoji = {
            "user_answer": "📝",
            "competitor": "🏢",
            "optimized": "✨"
        }.get(row["type"], "•")

        html += '<tr>'
        html += f'<td>{type_emoji} {row["label"]}</td>'
        html += f'<td>{row["type"]}</td>'

        # Score iniziale
        if row["initial_score"] is not None:
            score_class = get_score_class(row["initial_score"])
            html += f'<td><span class="score-badge {score_class}">{row["initial_score"]:.4f}</span></td>'
        else:
            html += '<td>-</td>'

        # Score iterazioni
        last_valid_score = row["initial_score"]
        for score in row["iteration_scores"]:
            if score is not None:
                score_class = get_score_class(score)
                html += f'<td><span class="score-badge {score_class}">{score:.4f}</span></td>'
                last_valid_score = score
            else:
                html += '<td>-</td>'

        # Variazione
        if num_iterations > 0 and row["initial_score"] is not None and last_valid_score is not None:
            variation = ((last_valid_score - row["initial_score"]) / row["initial_score"] * 100) if row["initial_score"] > 0 else 0
            color = "#10b981" if variation >= 0 else "#ef4444"
            html += f'<td style="color:{color}; font-weight:600;">{variation:+.2f}%</td>'
        elif num_iterations > 0:
            html += '<td>-</td>'

        html += '</tr>'

    html += '</tbody></table>'
    return html


# ==================== LOG CALLBACK CLASS ====================
class StreamlitLogCallback:
    """Callback che aggiorna i log in tempo reale"""
    def __init__(self, log_placeholder, status_placeholder):
        self.log_placeholder = log_placeholder
        self.status_placeholder = status_placeholder

    def __call__(self, log_entry: Dict):
        """Chiamato quando arriva un nuovo log"""
        st.session_state.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": log_entry.get("agent", "agent"),
            "level": log_entry.get("level", "info"),
            "message": log_entry.get("message", "")
        })

        # Aggiorna fase corrente
        msg = log_entry.get("message", "")
        if "FASE" in msg:
            st.session_state.current_phase = msg

        # Forza update UI
        self.update_display()

    def update_display(self):
        """Aggiorna display log"""
        self.log_placeholder.markdown(
            f'<div class="log-container">{render_logs_html()}</div>',
            unsafe_allow_html=True
        )
        if st.session_state.current_phase:
            self.status_placeholder.markdown(
                f'<div class="phase-indicator">{st.session_state.current_phase}</div>',
                unsafe_allow_html=True
            )


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image(MOCA_LOGO_URL, width=80)
    st.markdown("# 🚀 AI Overview Optimizer")
    st.markdown("**v2.1 - Real-time Logging**")

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

    available_models = {}
    if openai_key:
        available_models.update(OPENAI_MODELS)
    if gemini_key:
        available_models.update(GEMINI_MODELS)

    if not available_models:
        st.warning("Inserisci almeno una API key")
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
        placeholder="Inserisci qui la tua risposta attuale...",
        help="Se hai già un contenuto, inseriscilo qui",
        height=100
    )

    col_loc, col_lang = st.columns(2)
    with col_loc:
        location = st.selectbox("📍 Location", options=list(LOCATION_CODES.keys()), index=0)
    with col_lang:
        language = st.selectbox("🌐 Language", options=list(LANGUAGE_CODES.keys()), index=0)

    st.divider()

    # ===== PARAMETRI AVANZATI =====
    st.markdown("## 🎛️ Parametri Avanzati")

    max_iterations = st.slider(
        "🔄 Iterazioni",
        min_value=1, max_value=MAX_ITERATIONS, value=DEFAULT_ITERATIONS,
        help="Numero di cicli di ottimizzazione"
    )

    max_serp_results = st.slider(
        "📊 Risultati SERP",
        min_value=5, max_value=MAX_SERP_RESULTS, value=DEFAULT_SERP_RESULTS
    )

    max_sources = st.slider(
        "🌐 Fonti AIO",
        min_value=3, max_value=MAX_SOURCES, value=5
    )

    st.divider()

    # ===== VALIDAZIONE =====
    credentials_valid = all([dataforseo_login, dataforseo_password, openai_key])
    input_valid = bool(keyword)

    if not credentials_valid:
        st.warning("⚠️ Inserisci DataForSEO e OpenAI keys")
    elif not input_valid:
        st.info("📝 Inserisci la keyword")
    else:
        st.success("✅ Pronto")

    st.divider()

    # ===== BOTTONE ANALISI =====
    analyze_button = st.button(
        "🚀 Avvia Analisi",
        use_container_width=True,
        disabled=not (credentials_valid and input_valid) or st.session_state.running
    )

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 10])
with col1:
    st.image(MOCA_LOGO_URL, width=60)
with col2:
    st.title("🚀 AI Overview Content Optimizer v2.1")
    st.markdown("**Ottimizza contenuti per Google AI Overview** | by [Moca Interactive](https://mocainteractive.com)")

st.divider()

# ==================== MAIN AREA LAYOUT ====================
# Creiamo placeholder per log real-time e risultati
main_status = st.empty()
main_log = st.empty()
main_progress = st.empty()
main_results = st.container()

# ==================== ANALISI ====================
if analyze_button:
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.results = None
    st.session_state.current_phase = ""

    # Crea callback per log real-time
    log_callback = StreamlitLogCallback(main_log, main_status)

    add_log("🚀 Avvio workflow multi-agente...", level="info", agent="system")
    log_callback.update_display()

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

    # Progress bar
    progress_bar = main_progress.progress(0, text="Inizializzazione...")

    try:
        # Esegui workflow
        orchestrator = OrchestratorAgent(log_callback=log_callback)

        progress_bar.progress(10, text="📡 Recupero dati SERP...")
        log_callback.update_display()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run con aggiornamenti periodici
        async def run_with_updates():
            return await orchestrator.run(state)

        result_state = loop.run_until_complete(run_with_updates())
        loop.close()

        progress_bar.progress(100, text="✅ Completato!")
        add_log("✅ Workflow completato con successo!", level="success", agent="system")
        log_callback.update_display()

        st.session_state.results = result_state
        st.session_state.running = False

        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Errore: {str(e)}")
        add_log(f"❌ Errore: {str(e)}", level="error", agent="system")
        log_callback.update_display()
        st.session_state.running = False
        import traceback
        st.code(traceback.format_exc())

# ==================== DISPLAY RUNNING LOGS ====================
if st.session_state.running:
    main_status.markdown(
        f'<div class="phase-indicator">{st.session_state.current_phase or "Avvio..."}</div>',
        unsafe_allow_html=True
    )
    main_log.markdown(
        f'<div class="log-container">{render_logs_html()}</div>',
        unsafe_allow_html=True
    )

# ==================== RISULTATI ====================
if st.session_state.results and not st.session_state.running:
    state = st.session_state.results

    # ========== LOG FINALE ==========
    with st.expander("📋 Log Completo Esecuzione", expanded=False):
        st.markdown(
            f'<div class="log-container" style="max-height:600px">{render_logs_html()}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========== METRICHE OVERVIEW ==========
    st.header("📊 RISULTATI ANALISI")

    col1, col2, col3, col4 = st.columns(4)

    # Calcola score iniziale utente
    initial_user_score = 0
    if state.initial_ranking:
        user_item = next((r for r in state.initial_ranking if r["type"] == "user_answer"), None)
        if user_item:
            initial_user_score = user_item["score"]

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state.best_score:.1%}</div>
            <div class="metric-label">Score Finale</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if initial_user_score > 0:
            improvement = ((state.best_score - initial_user_score) / initial_user_score * 100)
        else:
            improvement = 0
        color = "#10b981" if improvement >= 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{improvement:+.1f}%</div>
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
            <div class="metric-value">{len(state.competitor_contents)}</div>
            <div class="metric-label">Competitor Analizzati</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== AI OVERVIEW ==========
    if state.ai_overview_text:
        st.markdown("---")
        st.subheader("🎯 AI Overview di Google (Riferimento)")
        st.success(state.ai_overview_text)

    # ========== COMPETITOR RESPONSES ==========
    if state.competitor_contents:
        st.markdown("---")
        st.subheader("🏢 Risposte dei Competitor (da AI Overview)")

        for i, comp in enumerate(state.competitor_contents):
            with st.expander(f"**{i+1}. {comp['domain']}** - {comp.get('title', '')[:50]}...", expanded=(i < 2)):
                st.markdown(f"**URL:** [{comp['url'][:60]}...]({comp['url']})")
                st.markdown(f"**Parole:** {comp['word_count']}")

                response_preview = comp.get('response_preview', comp['content'][:500])
                st.markdown("**Risposta del competitor:**")
                st.markdown(f"""
                <div class="competitor-response">
                {response_preview}
                </div>
                """, unsafe_allow_html=True)

                # Score del competitor
                comp_ranking = next((r for r in state.initial_ranking if comp['domain'] in r.get('label', '')), None)
                if comp_ranking:
                    st.metric("Score vs AI Overview", f"{comp_ranking['score']:.4f}")

    # ========== TABELLA EVOLUZIONE RANKING ==========
    st.markdown("---")
    st.subheader("📈 Evoluzione Ranking")

    st.markdown("""
    Questa tabella mostra come gli score cambiano ad ogni iterazione rispetto all'AI Overview di Google.
    """)

    ranking_table_html = create_ranking_evolution_table(state)
    st.markdown(ranking_table_html, unsafe_allow_html=True)

    # ========== ITERAZIONI DETTAGLIATE ==========
    if state.iterations:
        st.markdown("---")
        st.subheader("🔄 Dettaglio Iterazioni")

        for iteration in state.iterations:
            iter_num = iteration['iteration']
            score = iteration['score']
            improvement_pct = iteration.get('improvement', 0)

            color = "#10b981" if improvement_pct >= 0 else "#ef4444"

            st.markdown(f"""
            <div class="iteration-card">
                <h4>Iterazione {iter_num} - Score: {score:.4f} <span style="color:{color}">({improvement_pct:+.2f}%)</span></h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**💭 Ragionamento:**")
                reasoning = iteration.get("reasoning", "")
                st.info(reasoning if len(reasoning) < 600 else reasoning[:600] + "...")

            with col2:
                st.markdown("**📝 Risposta Ottimizzata:**")
                st.success(iteration.get("answer", ""))

            # Mini ranking per questa iterazione
            if iteration.get("ranking"):
                st.markdown("**📊 Ranking dopo questa iterazione:**")
                mini_ranking = []
                for r in iteration["ranking"][:5]:
                    mini_ranking.append({
                        "Pos": r["rank"],
                        "Fonte": r["label"],
                        "Score": f"{r['score']:.4f}"
                    })
                st.dataframe(pd.DataFrame(mini_ranking), use_container_width=True, hide_index=True)

            st.markdown("---")

    # ========== RISPOSTA FINALE ==========
    if state.best_answer:
        st.subheader("✨ RISPOSTA OTTIMIZZATA FINALE")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(state.best_answer)
        with col2:
            st.metric("Score Finale", f"{state.best_score:.4f}")
            st.metric("Parole", len(state.best_answer.split()))
            if initial_user_score > 0:
                st.metric("vs Iniziale", f"{((state.best_score - initial_user_score) / initial_user_score * 100):+.1f}%")

    # ========== ANALISI STRATEGICA ==========
    if state.strategic_analysis:
        st.markdown("---")
        st.subheader("📋 Analisi Strategica")

        with st.expander("Visualizza Analisi Completa", expanded=False):
            st.markdown("**Ragionamento:**")
            st.info(state.strategic_analysis.get("reasoning", ""))
            st.markdown("**Analisi:**")
            st.write(state.strategic_analysis.get("analysis", ""))

    # ========== PIANO CONTENUTO ==========
    if state.content_plan:
        st.markdown("---")
        st.subheader("📝 Piano Contenuto")

        with st.expander("Visualizza Piano Completo", expanded=False):
            st.markdown("**Ragionamento:**")
            st.info(state.content_plan.get("reasoning", ""))
            st.markdown("**Piano:**")
            st.write(state.content_plan.get("plan", ""))

    # ========== EXPORT ==========
    st.markdown("---")
    st.subheader("📥 Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        export_data = {
            "keyword": state.keyword,
            "best_answer": state.best_answer,
            "best_score": state.best_score,
            "initial_score": initial_user_score,
            "iterations": state.iterations,
            "initial_ranking": state.initial_ranking,
            "final_ranking": state.current_ranking,
            "ai_overview_text": state.ai_overview_text,
            "competitor_responses": [
                {"domain": c["domain"], "response": c.get("response_preview", ""), "score": next((r["score"] for r in state.initial_ranking if c["domain"] in r.get("label", "")), None)}
                for c in state.competitor_contents
            ],
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
        summary_data = {
            "Keyword": [state.keyword],
            "Score Iniziale": [f"{initial_user_score:.4f}"],
            "Score Finale": [f"{state.best_score:.4f}"],
            "Miglioramento": [f"{improvement:+.2f}%"],
            "Iterazioni": [len(state.iterations)],
            "Competitor": [len(state.competitor_contents)],
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
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Overview Optimization Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Keyword: {state.keyword}", ln=True)
            pdf.cell(0, 10, f"Score Iniziale: {initial_user_score:.4f}", ln=True)
            pdf.cell(0, 10, f"Score Finale: {state.best_score:.4f}", ln=True)
            pdf.cell(0, 10, f"Miglioramento: {improvement:+.2f}%", ln=True)
            pdf.cell(0, 10, f"Iterazioni: {len(state.iterations)}", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Risposta Ottimizzata:", ln=True)
            pdf.set_font("Arial", "", 10)
            # Handle encoding for PDF
            safe_answer = state.best_answer.encode('latin-1', 'replace').decode('latin-1') if state.best_answer else "N/A"
            pdf.multi_cell(0, 5, safe_answer[:2000])

            pdf_output = pdf.output(dest='S').encode('latin-1', 'replace')

            st.download_button(
                label="📑 Download PDF",
                data=pdf_output,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF export non disponibile: {e}")

# ==================== ISTRUZIONI INIZIALI ====================
if not st.session_state.results and not st.session_state.running:
    st.info("👈 **Per iniziare**: Configura le API keys e inserisci la keyword nella sidebar")

    st.markdown("""
    ## 🚀 Come Funziona v2.1

    ### Workflow Multi-Agente:

    1. 🔍 **SERP Analysis** - Recupera AI Overview e competitor
    2. 🕷️ **Content Scraping** - Estrae risposte dai competitor
    3. 📊 **Ranking Iniziale** - Calcola posizione vs AI Overview
    4. 🔄 **Ottimizzazione Iterativa** - Migliora progressivamente (ogni iterazione parte dalla precedente)
    5. 📋 **Analisi Strategica** - Intento, gap, opportunità
    6. 📝 **Piano Contenuto** - Struttura H1/H2/H3

    ### Novità v2.1:
    - ✅ **Log real-time** nel main content
    - ✅ **Tabella evoluzione ranking** per vedere i progressi
    - ✅ **Risposte competitor** visibili ed espandibili
    - ✅ **Iterazioni collegate** - ogni iterazione migliora la precedente
    """)

# ==================== FOOTER ====================
st.divider()
st.markdown(f"""
<div style='text-align: center; color: {MOCA_COLORS['gray']}; padding: 20px;'>
    <p>Sviluppato da <a href='https://mocainteractive.com' target='_blank' style='color: {MOCA_COLORS['primary']}; text-decoration: none;'><strong>Moca Interactive</strong></a></p>
    <p style='font-size: 0.8em;'>v2.1 - Multi-Agent System with Real-time Logging</p>
</div>
""", unsafe_allow_html=True)
