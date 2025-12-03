"""
AI Overview Content Optimizer v2.4
Real-time phase logging + Proper formatting + PDF Export
Developed by Moca Interactive
"""
import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict, List, Optional
import time
import io

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
if 'phase_logs' not in st.session_state:
    st.session_state.phase_logs = {}  # {phase_name: [logs]}
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'phase_placeholders' not in st.session_state:
    st.session_state.phase_placeholders = {}

# ==================== MINIMAL CSS (dark mode safe) ====================
st.markdown("""
<style>
    .phase-box {
        background-color: #161b22;
        color: #c9d1d9;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.8rem;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border: 1px solid #30363d;
    }
    .phase-title {
        color: #58a6ff;
        font-weight: bold;
        font-size: 1rem;
        margin-bottom: 8px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }
    .log-line { margin: 4px 0; line-height: 1.4; }
    .log-info { color: #8b949e; }
    .log-success { color: #3fb950; }
    .log-warning { color: #d29922; }
    .log-error { color: #f85149; }

    .answer-box {
        background-color: #0d1117;
        border: 1px solid #238636;
        border-radius: 8px;
        padding: 16px;
        white-space: pre-wrap;
        font-family: system-ui, -apple-system, sans-serif;
        line-height: 1.6;
    }

    .analysis-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        font-family: system-ui, -apple-system, sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def get_log_color(level: str) -> str:
    """Get color for log level"""
    colors = {
        "info": "#8b949e",
        "success": "#3fb950",
        "warning": "#d29922",
        "error": "#f85149"
    }
    return colors.get(level, '#8b949e')


def render_phase_logs(phase_name: str, logs: List[Dict], is_active: bool = False) -> str:
    """Render logs for a single phase"""
    status_icon = "⏳" if is_active else "✅"
    html = f'<div class="phase-box">'
    html += f'<div class="phase-title">{status_icon} {phase_name}</div>'

    for log in logs:
        color = get_log_color(log.get('level', 'info'))
        msg = log.get('message', '').replace('<', '&lt;').replace('>', '&gt;')
        html += f'<div class="log-line" style="color:{color}">• {msg}</div>'

    html += '</div>'
    return html


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
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)


def format_answer_html(answer: str) -> str:
    """Format answer preserving newlines"""
    if not answer:
        return ""
    # Escape HTML but preserve structure
    escaped = answer.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Convert newlines to <br>
    formatted = escaped.replace('\n\n', '</p><p>').replace('\n', '<br>')
    return f'<div class="answer-box"><p>{formatted}</p></div>'


def parse_json_safely(text: str) -> Optional[Dict]:
    """Try to parse JSON from text"""
    if not text:
        return None
    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except:
                    pass
        elif '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except:
                    pass
        # Try to find JSON object in text
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            try:
                return json.loads(text[start:end])
            except:
                pass
    return None


def render_strategic_analysis(analysis: Dict):
    """Render strategic analysis with proper formatting"""
    if not analysis:
        st.info("Non disponibile")
        return

    reasoning = analysis.get("reasoning", "")
    analysis_text = analysis.get("analysis", "")

    # Show reasoning
    if reasoning:
        st.markdown("**💭 Ragionamento:**")
        st.markdown(f"> {reasoning}")

    # Try to parse JSON
    parsed = parse_json_safely(analysis_text)

    if parsed:
        # Formatted JSON display
        if "INTENTO DI RICERCA" in str(parsed) or "intento" in str(parsed).lower():
            for key, value in parsed.items():
                st.markdown(f"**{key}**")
                if isinstance(value, dict):
                    for k, v in value.items():
                        st.markdown(f"- **{k}:** {v}")
                elif isinstance(value, list):
                    for item in value:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(f"{value}")
        else:
            st.json(parsed)
    else:
        # Plain text display with formatting
        st.markdown(analysis_text)


def render_content_plan(plan: Dict):
    """Render content plan with proper formatting"""
    if not plan:
        st.info("Non disponibile")
        return

    reasoning = plan.get("reasoning", "")
    plan_text = plan.get("plan", "")

    # Show reasoning
    if reasoning:
        st.markdown("**💭 Ragionamento:**")
        st.markdown(f"> {reasoning}")

    # Try to parse JSON
    parsed = parse_json_safely(plan_text)

    if parsed:
        # Formatted display
        if isinstance(parsed, dict):
            for section, content in parsed.items():
                with st.expander(f"📌 {section}", expanded=True):
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    st.markdown(f"**{k}:** {v}")
                                st.markdown("---")
                            else:
                                st.markdown(f"- {item}")
                    elif isinstance(content, dict):
                        for k, v in content.items():
                            st.markdown(f"**{k}:** {v}")
                    else:
                        st.markdown(str(content))
        else:
            st.json(parsed)
    else:
        # Plain text
        st.markdown(plan_text)


def generate_pdf_report(state: AgentState) -> Optional[bytes]:
    """Generate PDF report from results"""
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def safe_text(text: str) -> str:
            """Safely encode text for PDF"""
            if not text:
                return ""
            return text.encode('latin-1', 'replace').decode('latin-1')

        def new_line():
            """Move to new line"""
            pdf.set_xy(pdf.l_margin, pdf.get_y() + pdf.font_size * 1.5)

        # Title
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 15, 'AI Overview Optimizer Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        # Keyword
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'Keyword: {safe_text(state.keyword)}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        # AI Overview
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Google AI Overview:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Helvetica', '', 10)
        if state.ai_overview_text:
            pdf.multi_cell(0, 6, safe_text(state.ai_overview_text[:1500]))
        else:
            pdf.cell(0, 6, 'Non disponibile per questa keyword', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(8)

        # Best Answer
        pdf.set_font('Helvetica', 'B', 12)
        score_text = f'{state.best_score:.4f}' if state.best_score else 'N/A'
        pdf.cell(0, 8, f'Risposta Ottimizzata (Score: {score_text}):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Helvetica', '', 10)
        if state.best_answer:
            pdf.multi_cell(0, 6, safe_text(state.best_answer[:2000]))
        else:
            pdf.cell(0, 6, 'Non disponibile', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(8)

        # Iterations Summary
        if state.iterations:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Riepilogo Iterazioni:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Helvetica', '', 10)
            for it in state.iterations:
                imp = it.get('improvement', 0)
                pdf.cell(0, 6, f"Iterazione {it['iteration']}: Score {it['score']:.4f} ({imp:+.2f}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(8)

        # Ranking
        if state.current_ranking:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Ranking Finale:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Helvetica', '', 10)
            for r in state.current_ranking[:5]:
                label = safe_text(r.get('label', 'N/A'))
                pdf.cell(0, 6, f"#{r['rank']} - {label}: {r['score']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Output as bytes
        pdf_output = pdf.output()

        if isinstance(pdf_output, (bytes, bytearray)):
            return bytes(pdf_output)
        return None

    except Exception as e:
        import traceback
        print(f"PDF Error: {e}")
        print(traceback.format_exc())
        return None


# ==================== LOAD SECRETS ====================
def get_secret(key: str, default: str = "") -> str:
    """Get secret from Streamlit secrets or return default"""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# Pre-load secrets
DEFAULT_DFS_LOGIN = get_secret("DATAFORSEO_LOGIN", "")
DEFAULT_DFS_PASSWORD = get_secret("DATAFORSEO_PASSWORD", "")
DEFAULT_OPENAI_KEY = get_secret("OPENAI_API_KEY", "")
DEFAULT_GCP_PROJECT = get_secret("GOOGLE_PROJECT_ID", "")
DEFAULT_GCP_CREDS = get_secret("GOOGLE_CREDENTIALS_JSON", "")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image(MOCA_LOGO_URL, width=80)
    st.markdown("## AI Overview Optimizer")

    st.divider()

    # API KEYS
    with st.expander("🔑 API Keys", expanded=not bool(DEFAULT_OPENAI_KEY)):
        dataforseo_login = st.text_input("DataForSEO Login", value=DEFAULT_DFS_LOGIN, key="dfs_login")
        dataforseo_password = st.text_input("DataForSEO Password", type="password",
                                            value=DEFAULT_DFS_PASSWORD, key="dfs_pass")
        st.markdown("---")
        openai_key = st.text_input("OpenAI API Key", type="password",
                                   value=DEFAULT_OPENAI_KEY, key="oai_key")
        st.markdown("---")
        gemini_key = st.text_input("Gemini Key (opz.)", type="password", key="gem_key")

    # Google Cloud (optional)
    with st.expander("☁️ Google Cloud Ranking", expanded=bool(DEFAULT_GCP_PROJECT)):
        google_project_id = st.text_input("Project ID", value=DEFAULT_GCP_PROJECT, key="gcp_project")
        google_credentials = st.text_area("Service Account JSON", height=100,
                                          value=DEFAULT_GCP_CREDS, key="gcp_creds",
                                          help="Incolla il JSON delle credenziali service account")
        use_google_ranking = st.checkbox("Usa Google Ranking API", key="use_google_ranking",
                                         value=bool(DEFAULT_GCP_PROJECT),
                                         help="Usa Discovery Engine per reranking semantico")

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

    analyze_btn = st.button("🚀 Avvia Analisi", width="stretch",
                            disabled=not ready or st.session_state.running)

# ==================== MAIN ====================
st.title("🚀 AI Overview Optimizer")
st.caption("by [Moca Interactive](https://mocainteractive.com)")
st.divider()

# ==================== EXECUTION ====================
if analyze_btn:
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.phase_logs = {}
    st.session_state.current_phase = None
    st.session_state.results = None

    # Create placeholders for each phase
    phase_containers = {}
    phase_names = [
        "FASE 1: Recupero dati SERP",
        "FASE 2: Scraping Contenuti",
        "FASE 3: Ranking Iniziale",
        "FASE 4: Ciclo Ottimizzazione",
        "FASE 5: Analisi Strategica",
        "FASE 6: Piano Contenuto"
    ]

    for phase in phase_names:
        phase_containers[phase] = st.empty()

    def detect_phase(message: str) -> Optional[str]:
        """Detect which phase a log message belongs to"""
        for phase in phase_names:
            # Check for phase marker
            phase_key = phase.split(":")[0]  # "FASE 1", "FASE 2", etc.
            if phase_key in message or phase.split(": ")[1] in message:
                return phase
        return None

    # Use session state instead of nonlocal
    st.session_state.current_phase = None

    def log_callback(entry: Dict):
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": entry.get("level", "info"),
            "message": entry.get("message", "")
        }

        msg = log_entry["message"]

        # Check if this starts a new phase
        detected_phase = detect_phase(msg)
        if detected_phase:
            st.session_state.current_phase = detected_phase
            if st.session_state.current_phase not in st.session_state.phase_logs:
                st.session_state.phase_logs[st.session_state.current_phase] = []

        # Add to current phase logs
        if st.session_state.current_phase:
            st.session_state.phase_logs[st.session_state.current_phase].append(log_entry)

            # Update the phase display
            if st.session_state.current_phase in phase_containers:
                phase_containers[st.session_state.current_phase].markdown(
                    render_phase_logs(
                        st.session_state.current_phase,
                        st.session_state.phase_logs[st.session_state.current_phase],
                        is_active=True
                    ),
                    unsafe_allow_html=True
                )

        # Also store in global logs
        st.session_state.logs.append(log_entry)

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
        dataforseo_password=dataforseo_password,
        google_project_id=google_project_id if 'google_project_id' in dir() else "",
        google_credentials_json=google_credentials if 'google_credentials' in dir() else "",
        ranking_method="google" if ('use_google_ranking' in dir() and use_google_ranking) else "embeddings"
    )

    try:
        orchestrator = OrchestratorAgent(log_callback=log_callback)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_state = loop.run_until_complete(orchestrator.run(state))
        loop.close()

        st.session_state.results = result_state
        st.session_state.running = False

        # Mark all phases as complete
        for phase in phase_names:
            if phase in st.session_state.phase_logs and phase in phase_containers:
                phase_containers[phase].markdown(
                    render_phase_logs(phase, st.session_state.phase_logs[phase], is_active=False),
                    unsafe_allow_html=True
                )

        st.rerun()

    except Exception as e:
        st.error(f"Errore: {e}")
        st.session_state.running = False
        import traceback
        st.code(traceback.format_exc())

# ==================== RESULTS ====================
if st.session_state.results:
    state = st.session_state.results

    # Show execution logs in expander
    with st.expander("📋 Log Esecuzione Completo", expanded=False):
        for phase_name, logs in st.session_state.phase_logs.items():
            st.markdown(render_phase_logs(phase_name, logs, is_active=False), unsafe_allow_html=True)

    st.divider()

    # ===== AI OVERVIEW =====
    if getattr(state, 'synthetic_reference', False):
        st.subheader("🤖 Riferimento Sintetico (AI Overview non disponibile)")
        st.warning("Google non ha restituito un AI Overview per questa keyword. È stato generato un riferimento sintetico basato sui competitor.")
        if state.ai_overview_text:
            st.markdown(format_answer_html(state.ai_overview_text), unsafe_allow_html=True)
            st.caption(f"{len(state.ai_overview_text)} caratteri | Generato da LLM")
    else:
        st.subheader("🎯 AI Overview di Google")
        if state.ai_overview_text:
            st.markdown(format_answer_html(state.ai_overview_text), unsafe_allow_html=True)
            st.caption(f"{len(state.ai_overview_text)} caratteri | {len(state.ai_overview_sources)} fonti")
        else:
            st.warning("AI Overview non disponibile per questa keyword")

    st.divider()

    # ===== COMPETITOR =====
    st.subheader("🏢 Risposte Competitor")
    if state.competitor_contents:
        for i, comp in enumerate(state.competitor_contents):
            with st.expander(f"{i+1}. {comp['domain']}", expanded=(i < 2)):
                st.caption(f"URL: {comp['url']}")
                preview = comp.get('response_preview', comp['content'][:500])
                st.markdown(format_answer_html(preview), unsafe_allow_html=True)
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
            improvement = it.get('improvement', 0)

            # Color based on improvement
            if improvement > 0:
                score_color = "#3fb950"  # green
            elif improvement < 0:
                score_color = "#f85149"  # red
            else:
                score_color = "#8b949e"  # gray

            st.markdown(f"### Iterazione {iter_num}")
            st.markdown(f"**Score:** <span style='color:{score_color}'>{score:.4f} ({improvement:+.2f}%)</span>",
                       unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**💭 Ragionamento:**")
                reasoning = it.get("reasoning", "N/A")
                st.markdown(f"> {reasoning}")

            with col2:
                st.markdown("**📝 Risposta:**")
                answer = it.get("answer", "N/A")
                st.markdown(format_answer_html(answer), unsafe_allow_html=True)

            # Ranking dopo questa iterazione
            if it.get("ranking"):
                render_ranking_table(it["ranking"], f"Ranking dopo iterazione {iter_num}")

            st.divider()
    else:
        st.info("Nessuna iterazione")

    # ===== RISPOSTA FINALE =====
    st.subheader("✨ Risposta Ottimizzata Finale")
    if state.best_answer:
        st.markdown(format_answer_html(state.best_answer), unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score Finale", f"{state.best_score:.4f}")
        with col2:
            st.metric("Parole", len(state.best_answer.split()))
        with col3:
            st.metric("Caratteri", len(state.best_answer))
    else:
        st.warning("Nessuna risposta generata")

    st.divider()

    # ===== ANALISI STRATEGICA =====
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Analisi Strategica")
        render_strategic_analysis(state.strategic_analysis)

    with col2:
        st.subheader("📝 Piano Contenuto")
        render_content_plan(state.content_plan)

    st.divider()

    # ===== EXPORT =====
    st.subheader("📥 Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        export_data = {
            "keyword": state.keyword,
            "best_answer": state.best_answer,
            "best_score": state.best_score,
            "iterations": state.iterations,
            "ai_overview_text": state.ai_overview_text,
            "strategic_analysis": state.strategic_analysis,
            "content_plan": state.content_plan,
            "initial_ranking": state.initial_ranking,
            "generated_at": datetime.now().isoformat()
        }
        st.download_button(
            "📄 Scarica JSON",
            json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            width="stretch"
        )

    with col2:
        csv_data = pd.DataFrame([{
            "Keyword": state.keyword,
            "Score Finale": state.best_score,
            "Iterazioni": len(state.iterations),
            "Risposta": state.best_answer[:500] if state.best_answer else ""
        }]).to_csv(index=False)
        st.download_button(
            "📊 Scarica CSV",
            csv_data,
            "summary.csv",
            "text/csv",
            width="stretch"
        )

    with col3:
        try:
            pdf_bytes = generate_pdf_report(state)
            if pdf_bytes and isinstance(pdf_bytes, (bytes, bytearray)):
                st.download_button(
                    "📕 Scarica PDF",
                    data=pdf_bytes,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
            else:
                st.button("📕 PDF non disponibile", disabled=True, width="stretch")
        except Exception as e:
            st.button("📕 Errore PDF", disabled=True, width="stretch")

# ==================== INSTRUCTIONS ====================
if not st.session_state.results and not st.session_state.running:
    st.info("👈 Configura API keys e keyword per iniziare")

    st.markdown("""
    ### Workflow
    1. 🎯 Recupera AI Overview da Google
    2. 🏢 Scraping risposte competitor
    3. 📊 Ranking iniziale (similarità vs AI Overview)
    4. 🔄 Ottimizzazione iterativa
    5. 📋 Analisi strategica
    6. 📝 Piano contenuto

    ### Note
    - Lo score rappresenta la **similarità** con l'AI Overview di Google (0-1)
    - Score più alto = contenuto più simile a quello che Google mostra
    - Il sistema usa embeddings OpenAI per calcolare la similarità semantica
    """)

# ==================== FOOTER ====================
st.divider()
st.caption("Moca Interactive | AI Overview Optimizer v2.4")
