"""
AI Overview Content Optimizer - Streamlit Interface
Developed by Moca Interactive
"""
import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.optimization_controller import OptimizationController
from utils.logger import logger
from config import MOCA_COLORS

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Overview Content Optimizer | Moca",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    .stProgress > div > div {{
        background-color: {MOCA_COLORS['primary']};
    }}

    .sidebar .sidebar-content {{
        background-color: {MOCA_COLORS['secondary']};
    }}

    .stAlert {{
        border-left: 4px solid {MOCA_COLORS['primary']};
    }}

    [data-testid="stMetricValue"] {{
        color: {MOCA_COLORS['primary']};
        font-size: 2rem;
        font-weight: 700;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {MOCA_COLORS['secondary']};
        color: #191919;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {MOCA_COLORS['primary']};
        color: white;
    }}

    .entity-tag {{
        display: inline-block;
        background-color: {MOCA_COLORS['secondary']};
        color: {MOCA_COLORS['dark']};
        padding: 4px 12px;
        border-radius: 16px;
        margin: 4px;
        font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2 = st.columns([1, 10])
with col1:
    st.image(
        "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1-192x192.png",
        width=60
    )
with col2:
    st.title("🔍 AI Overview Content Optimizer")
    st.markdown("**Ottimizza contenuti per Google AI Overview** | by [Moca Interactive](https://mocainteractive.com)")

st.divider()

# ==================== SIDEBAR - CREDENZIALI ====================
with st.sidebar:
    st.image(
        "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1-192x192.png",
        width=80
    )
    st.markdown("## 🔐 Configurazione")

    # ===== SEZIONE API KEYS =====
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
            help="Formato: sk-...",
            key="openai_key"
        )

        st.markdown("---")
        st.markdown("### Jina AI (Reranker)")
        jina_key = st.text_input(
            "API Key",
            type="password",
            help="Formato: jina_...",
            key="jina_key"
        )

    # ===== VALIDAZIONE CREDENZIALI =====
    credentials_valid = all([
        dataforseo_login,
        dataforseo_password,
        openai_key,
        jina_key
    ])

    if not credentials_valid:
        st.warning("⚠️ Inserisci tutte le API keys per continuare")
    else:
        st.success("✅ Credenziali configurate")

    st.divider()

    # ===== PARAMETRI ANALISI =====
    st.markdown("## ⚙️ Parametri Analisi")

    target_url = st.text_input(
        "🔗 URL Target",
        placeholder="https://example.com/article",
        help="URL della pagina da ottimizzare"
    )

    keyword = st.text_input(
        "🔍 Keyword",
        placeholder="come rinnovare il passaporto",
        help="Keyword principale da analizzare"
    )

    col_loc, col_lang = st.columns(2)
    with col_loc:
        location = st.selectbox(
            "📍 Location",
            ["Italy", "United States", "United Kingdom", "Germany", "France", "Spain"],
            index=0
        )

    with col_lang:
        language = st.selectbox(
            "🌐 Language",
            ["Italian", "English", "German", "French", "Spanish"],
            index=0
        )

    max_sources = st.slider(
        "📊 Max Sources",
        min_value=3,
        max_value=10,
        value=5,
        help="Numero massimo di fonti competitor da analizzare"
    )

    st.divider()

    # ===== INFO COSTI =====
    with st.expander("💰 Stima Costi"):
        st.markdown(f"""
        **Costo stimato per questa analisi:**
        - DataForSEO: ~$0.10
        - OpenAI: ~$0.05
        - Jina: Gratuito

        **Totale**: ~$0.15-0.20

        **Fonti**: {max_sources}
        """)

    st.divider()

    # ===== BOTTONE ANALISI =====
    analyze_button = st.button(
        "🚀 Avvia Analisi",
        use_container_width=True,
        disabled=not (credentials_valid and target_url and keyword)
    )

# ==================== ISTRUZIONI INIZIALI ====================
if not credentials_valid:
    st.info("👈 **Per iniziare**: Inserisci le tue API keys nella sidebar")

    with st.expander("📖 Come ottenere le API Keys"):
        st.markdown("""
        ### 🔑 DataForSEO
        1. Registrati su [DataForSEO](https://dataforseo.com/)
        2. Vai su Dashboard → API Access
        3. Copia Login e Password

        ### 🔑 OpenAI
        1. Vai su [OpenAI Platform](https://platform.openai.com/)
        2. Account → API Keys
        3. Crea nuova key (sk-...)

        ### 🔑 Jina AI
        1. Registrati su [Jina AI](https://jina.ai/)
        2. Dashboard → API Keys
        3. Crea nuova key (jina_...)
        """)

    st.stop()

if not (target_url and keyword):
    st.info("👈 **Inserisci URL e Keyword** nella sidebar per continuare")

    # ========== LANDING PAGE ==========
    st.markdown("""
    ## 🚀 Come Funziona

    Questo strumento analizza i tuoi contenuti e li ottimizza per aumentare le probabilità di comparire negli **AI Overview di Google**.

    ### Workflow:

    1. 🔍 **SERP Analysis** - Recupera AI Overview e competitor data
    2. 🕷️ **Content Scraping** - Estrae contenuti dalle fonti top
    3. 📊 **Relevance Scoring** - Calcola rilevanza contestuale con AI reranker
    4. 🎯 **Gap Analysis** - Identifica entità e concetti mancanti
    5. ✨ **Optimization** - Genera versione ottimizzata del contenuto
    6. 💡 **Recommendations** - Fornisce suggerimenti actionable
    """)

    st.info("""
    💡 **Tip**: Per risultati ottimali, scegli keyword informative che già mostrano AI Overview nelle SERP.
    Esempio: "come fare...", "cosa significa...", "guida a..."
    """)

    st.stop()

# ==================== ANALISI ====================
if analyze_button:
    try:
        # Initialize controller with user credentials
        controller = OptimizationController(
            dataforseo_login=dataforseo_login,
            dataforseo_password=dataforseo_password,
            openai_api_key=openai_key,
            jina_api_key=jina_key,
            reranker_provider="jina"
        )

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run analysis
        status_text.text("🔍 Analisi in corso...")
        progress_bar.progress(10)

        results = controller.optimize_content(
            target_url=target_url,
            keyword=keyword,
            location=location,
            language=language,
            max_sources=max_sources
        )

        progress_bar.progress(100)
        status_text.text("✅ Analisi completata!")

        # Check for errors
        if not results.get("success", False):
            st.error(f"❌ Errore: {results.get('error', 'Errore sconosciuto')}")
            st.stop()

        # Store results in session state
        st.session_state['results'] = results

    except Exception as e:
        st.error(f"❌ Errore durante l'analisi: {str(e)}")
        logger.error(f"Analisi fallita: {e}")
        st.stop()

# ==================== VISUALIZZAZIONE RISULTATI ====================
if 'results' in st.session_state:
    results = st.session_state['results']

    st.markdown("---")
    st.header("📊 RISULTATI ANALISI")

    # ========== 1. OVERVIEW METRICS ==========
    analysis = results.get("analysis", {})
    target_content = results.get("target_content", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Relevance Score Target",
            f"{analysis.get('current_relevance_score', 0):.3f}"
        )

    with col2:
        is_source = "✅ Sì" if analysis.get("is_ai_overview_source") else "❌ No"
        st.metric(
            "In AI Overview",
            is_source
        )

    with col3:
        st.metric(
            "Fonti AI Overview",
            analysis.get("ai_overview_sources_count", 0)
        )

    with col4:
        st.metric(
            "Parole Target",
            target_content.get("word_count", 0)
        )

    # ========== 2. TARGET ANALYSIS ==========
    st.markdown("---")
    st.subheader("🎯 Analisi URL Target")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Risposta Estratta**")
        answer = target_content.get("answer", "N/A")
        st.info(answer[:500] + "..." if len(answer) > 500 else answer)

    with col2:
        st.markdown("**Metriche**")
        st.write(f"📝 Parole: **{target_content.get('word_count', 0)}**")
        st.write(f"📄 Caratteri: **{target_content.get('char_count', 0)}**")

    # ========== 3. TOP SOURCES ==========
    st.markdown("---")
    st.subheader("🏆 Top Competitor Sources")

    top_sources = results.get("top_sources", [])

    if top_sources:
        sources_df = pd.DataFrame([
            {
                "Domain": s.get("domain", "N/A"),
                "Relevance Score": f"{s.get('relevance_score', 0):.3f}",
                "Semantic Similarity": f"{s.get('semantic_similarity', 0):.3f}",
            }
            for s in top_sources[:5]
        ])

        st.dataframe(sources_df, use_container_width=True)

        with st.expander("📄 Visualizza risposte competitor"):
            for i, source in enumerate(top_sources[:3]):
                st.markdown(f"**{i+1}. {source.get('domain', 'N/A')}** (Score: {source.get('relevance_score', 0):.3f})")
                st.caption(source.get("answer_preview", "N/A"))
                st.markdown("---")
    else:
        st.warning("Nessuna fonte competitor trovata")

    # ========== 4. GAP ANALYSIS ==========
    st.markdown("---")
    st.subheader("🔍 Gap Analysis - Entità Mancanti")

    gap_analysis = results.get("gap_analysis", {})
    missing_entities = gap_analysis.get("missing_entities", [])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(
            "Entity Coverage",
            f"{gap_analysis.get('entity_coverage', 0):.1%}"
        )

    with col2:
        st.metric(
            "Similarity Media",
            f"{gap_analysis.get('semantic_similarity_avg', 0):.3f}"
        )

    if missing_entities:
        st.markdown("**🏷️ Entità da Aggiungere**")
        tags_html = " ".join([
            f'<span class="entity-tag">{ent.get("entity", "N/A")} ({ent.get("frequency", 0)}x)</span>'
            for ent in missing_entities[:10]
        ])
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.success("✅ Nessuna entità significativa mancante!")

    # ========== 5. OPTIMIZATION ==========
    st.markdown("---")
    st.subheader("✨ Risposta Ottimizzata")

    optimization = results.get("optimized_answer", {})

    if optimization:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Score Precedente",
                f"{analysis.get('current_relevance_score', 0):.3f}"
            )

        with col2:
            st.metric(
                "Score Ottimizzato",
                f"{optimization.get('new_relevance_score', 0):.3f}"
            )

        with col3:
            improvement = optimization.get('improvement_percentage', 0)
            st.metric(
                "Miglioramento",
                f"{improvement:+.1f}%"
            )

        st.markdown("**📝 Versione Ottimizzata**")
        st.success(optimization.get("text", "N/A"))

        with st.expander("🔄 Confronto Versioni"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Versione Originale**")
                answer = target_content.get("answer", "N/A")
                st.info(answer[:500] + "..." if len(answer) > 500 else answer)

            with col2:
                st.markdown("**Versione Ottimizzata**")
                st.success(optimization.get("text", "N/A"))
    else:
        st.warning("⚠️ Impossibile generare ottimizzazione")

    # ========== 6. FAN-OUT OPPORTUNITIES ==========
    fan_out = results.get("fan_out_opportunities", [])
    if fan_out:
        st.markdown("---")
        st.subheader("🔗 Opportunità Fan-Out Queries")

        fan_out_df = pd.DataFrame([
            {
                "Query": f.get("query", "N/A"),
                "Opportunità": f.get("opportunity_score", "N/A")
            }
            for f in fan_out[:5]
        ])

        st.dataframe(fan_out_df, use_container_width=True)

    # ========== 7. RECOMMENDATIONS ==========
    st.markdown("---")
    st.subheader("💡 Raccomandazioni")

    recommendations = results.get("recommendations", [])

    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ Il contenuto è già ben ottimizzato!")

    # ========== 8. EXPORT DATA ==========
    st.markdown("---")
    st.subheader("📥 Export Dati")

    col1, col2 = st.columns(2)

    with col1:
        json_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"ai_overview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        summary_data = {
            "URL Target": [target_url],
            "Keyword": [keyword],
            "Relevance Score": [analysis.get('current_relevance_score', 0)],
            "Optimized Score": [optimization.get('new_relevance_score', 0) if optimization else 0],
            "Improvement %": [optimization.get('improvement_percentage', 0) if optimization else 0],
            "Missing Entities": [len(missing_entities)],
            "Timestamp": [datetime.now().isoformat()]
        }
        summary_df = pd.DataFrame(summary_data)

        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV Summary",
            data=csv,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ==================== FOOTER ====================
st.divider()
st.markdown(f"""
<div style='text-align: center; color: {MOCA_COLORS['gray']}; padding: 20px;'>
    <p>Sviluppato da <a href='https://mocainteractive.com' target='_blank' style='color: {MOCA_COLORS['primary']}; text-decoration: none;'><strong>Moca Interactive</strong></a></p>
    <p style='font-size: 0.9em;'>© 2025 Moca Interactive. Tutti i diritti riservati.</p>
</div>
""", unsafe_allow_html=True)
