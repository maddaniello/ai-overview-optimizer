"""
AI Overview Content Optimizer - Streamlit Interface
Developed by Moca Interactive
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.optimization_controller import OptimizationController
from utils.logger import logger
import json

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Overview Content Optimizer | Moca",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Figtree', sans-serif;
    }
    
    .main {
        background-color: #FFFFFF;
    }
    
    h1, h2, h3 {
        color: #E52217 !important;
        font-weight: 700;
    }
    
    .stButton>button {
        background-color: #E52217;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
    }
    
    .stButton>button:hover {
        background-color: #c41d13;
    }
    
    .stProgress > div > div {
        background-color: #E52217;
    }
    
    .sidebar .sidebar-content {
        background-color: #FFE7E6;
    }
    
    .css-1d391kg {
        background-color: #FFE7E6;
    }
    
    .stAlert {
        border-left: 4px solid #E52217;
    }
    
    /* Custom metric styling */
    [data-testid="stMetricValue"] {
        color: #E52217;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #FFE7E6;
        color: #191919;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E52217;
        color: white;
    }
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
    st.stop()

# ==================== ANALISI ====================
if analyze_button:
    
    # Store credentials in session state for this analysis
    st.session_state['temp_credentials'] = {
        'dataforseo_login': dataforseo_login,
        'dataforseo_password': dataforseo_password,
        'openai_key': openai_key,
        'jina_key': jina_key,
        'location': location,
        'language': language
    }
    
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
        
        # Step 1: SERP Analysis
        status_text.text("🔍 Step 1/7: Recupero AI Overview da Google...")
        progress_bar.progress(10)
        
        # Step 2-7: ... (resto del codice di analisi come prima)
        
        # IMPORTANTE: Passa le credenziali al controller
        result = controller.optimize_content(
            target_url=target_url,
            keyword=keyword,
            location=location,
            language=language,
            max_sources=max_sources
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analisi completata!")
        
        # ... resto del codice per mostrare i risultati
        
    except Exception as e:
        st.error(f"❌ Errore durante l'analisi: {str(e)}")
        logger.error(f"Analisi fallita: {e}")
        st.stop()
    
    finally:
        # Clear temporary credentials from memory
        if 'temp_credentials' in st.session_state:
            del st.session_state['temp_credentials']

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #8A8A8A; padding: 20px;'>
    <p>Sviluppato da <a href='https://mocainteractive.com' target='_blank' style='color: #E52217; text-decoration: none;'><strong>Moca Interactive</strong></a></p>
    <p style='font-size: 0.9em;'>© 2025 Moca Interactive. Tutti i diritti riservati.</p>
</div>
""", unsafe_allow_html=True)

# ========== MAIN CONTENT ==========

if analyze_button:
    # Validazione input
    if not target_url or not keyword:
        st.error("❌ Inserisci URL e Keyword per iniziare")
        st.stop()
    
    # Inizializza controller
    try:
        controller = OptimizationController()
        st.success("✅ Controller inizializzato correttamente")
    except Exception as e:
        st.error(f"❌ Errore inizializzazione: {str(e)}")
        st.info("Verifica che le API keys siano configurate correttamente nel file .env")
        st.stop()
    
    # Container per progress
    progress_container = st.container()
    
    with progress_container:
        st.markdown(f"""
        <div class="step-header">
            🎯 ANALISI IN CORSO
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Esegui analisi
    try:
        # Crea event loop se necessario
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # STEP 1: SERP Data
        status_text.text("🔍 Step 1/8: Recupero SERP e AI Overview...")
        progress_bar.progress(12)
        
        results = loop.run_until_complete(
            controller.optimize_content(
                target_url=target_url,
                keyword=keyword,
                location=location,
                language=language,
                max_sources=max_sources,
                include_fan_out=include_fan_out
            )
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analisi completata!")
        
        # ========== VISUALIZZAZIONE RISULTATI ==========
        
        st.markdown("---")
        st.header("📊 RISULTATI ANALISI")
        
        # ========== 1. OVERVIEW METRICS ==========
        st.markdown("""
        <div class="step-header">
            📈 METRICHE GENERALI
        </div>
        """, unsafe_allow_html=True)
        
        serp_data = results["serp_data"]
        target_analysis = results["target_analysis"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Relevance Score Target",
                f"{target_analysis.get('relevance_score', 0):.3f}",
                delta=None
            )
        
        with col2:
            has_ai_overview = "✅ Presente" if serp_data["has_ai_overview"] else "❌ Assente"
            st.metric(
                "AI Overview",
                has_ai_overview,
                delta=None
            )
        
        with col3:
            sources_count = len(serp_data["ai_overview_sources"]) if serp_data["has_ai_overview"] else 0
            st.metric(
                "Fonti AI Overview",
                sources_count,
                delta=None
            )
        
        with col4:
            st.metric(
                "Risultati Totali",
                serp_data["total_results"],
                delta=None
            )
        
        # ========== 2. TARGET ANALYSIS ==========
        st.markdown("---")
        st.markdown("""
        <div class="step-header">
            🎯 ANALISI URL TARGET
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Risposta Estratta")
            st.info(target_analysis["answer"])
        
        with col2:
            st.subheader("Metriche Qualità")
            quality = target_analysis["quality_metrics"]
            st.write(f"📝 Parole: **{quality['word_count']}**")
            st.write(f"📄 Caratteri: **{quality['char_count']}**")
            st.write(f"💬 Frasi: **{quality['sentence_count']}**")
            st.write(f"📊 Parole/frase: **{quality['avg_words_per_sentence']}**")
        
        # ========== 3. COMPETITORS ANALYSIS ==========
        st.markdown("---")
        st.markdown("""
        <div class="step-header">
            🏆 TOP COMPETITOR SOURCES
        </div>
        """, unsafe_allow_html=True)
        
        sources_analysis = results["sources_analysis"]
        
        if sources_analysis:
            # Tabella comparativa
            sources_df = pd.DataFrame([
                {
                    "Domain": s["domain"],
                    "Relevance Score": f"{s['relevance_score']:.3f}",
                    "Semantic Similarity": f"{s['semantic_similarity']:.3f}",
                    "Position": s.get("position", "-")
                }
                for s in sources_analysis[:5]
            ])
            
            st.dataframe(sources_df, use_container_width=True)
            
            # Dettagli per ogni source
            with st.expander("📄 Visualizza risposte competitor"):
                for i, source in enumerate(sources_analysis[:3]):
                    st.markdown(f"**{i+1}. {source['domain']}** (Score: {source['relevance_score']:.3f})")
                    st.caption(source["answer"])
                    st.markdown("---")
        else:
            st.warning("Nessuna fonte competitor trovata")
        
        # ========== 4. GAP ANALYSIS ==========
        st.markdown("---")
        st.markdown("""
        <div class="step-header">
            🔍 GAP ANALYSIS - Entità Mancanti
        </div>
        """, unsafe_allow_html=True)
        
        gap_analysis = results["gap_analysis"]
        missing_entities = gap_analysis["missing_entities"]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric(
                "Entità Target",
                gap_analysis["target_entities_count"]
            )
            st.metric(
                "Entità Competitor",
                gap_analysis["competitor_entities_count"]
            )
        
        with col2:
            coverage = gap_analysis["entity_coverage"]
            st.metric(
                "Entity Coverage",
                f"{coverage:.1%}",
                delta=f"{(coverage - 1) * 100:.1f}%" if coverage < 1 else None,
                delta_color="normal"
            )
        
        if missing_entities:
            st.subheader("🏷️ Entità da Aggiungere")
            
            # Visualizza come tags
            tags_html = " ".join([
                f'<span class="entity-tag">{ent["entity"]} ({ent["frequency"]}x)</span>'
                for ent in missing_entities
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.success("✅ Nessuna entità significativa mancante!")
        
        # ========== 5. OPTIMIZATION ==========
        st.markdown("---")
        st.markdown("""
        <div class="step-header">
            ✨ RISPOSTA OTTIMIZZATA
        </div>
        """, unsafe_allow_html=True)
        
        optimization = results["optimization"]
        
        if optimization:
            # Metriche miglioramento
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: {moca_branding.gray};">Score Precedente</div>
                    <div class="score-big">{optimization['previous_score']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: {moca_branding.gray};">Score Ottimizzato</div>
                    <div class="score-big">{optimization['new_relevance_score']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                improvement = optimization['improvement_percentage']
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: {moca_branding.gray};">Miglioramento</div>
                    <div class="score-big {improvement_class}">{improvement:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Testo ottimizzato
            st.subheader("📝 Versione Ottimizzata")
            st.success(optimization["optimized_answer"])
            
            # Comparazione affiancata
            with st.expander("🔄 Confronto Versioni"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Versione Originale**")
                    st.info(target_analysis["answer"])
                
                with col2:
                    st.markdown("**Versione Ottimizzata**")
                    st.success(optimization["optimized_answer"])
        else:
            st.warning("⚠️ Impossibile generare ottimizzazione (verifica OpenAI API key)")
        
        # ========== 6. FAN-OUT ANALYSIS ==========
        if include_fan_out and results.get("fan_out_analysis"):
            st.markdown("---")
            st.markdown("""
            <div class="step-header">
                🔗 FAN-OUT QUERIES - Opportunità
            </div>
            """, unsafe_allow_html=True)
            
            fan_out = results["fan_out_analysis"]
            
            if fan_out:
                fan_out_df = pd.DataFrame([
                    {
                        "Query": f["query"],
                        "Posizionato": "✅" if f["is_ranking"] else "❌",
                        "Posizione": f.get("position", "-"),
                        "Opportunità": "🎯 Alta" if f["opportunity"] else "✓"
                    }
                    for f in fan_out
                ])
                
                st.dataframe(fan_out_df, use_container_width=True)
                
                opportunities = [f for f in fan_out if f["opportunity"]]
                if opportunities:
                    st.info(f"💡 {len(opportunities)} opportunità di posizionamento rilevate!")
            else:
                st.info("Nessuna fan-out query trovata per questa keyword")
        
        # ========== 7. RECOMMENDATIONS ==========
        st.markdown("---")
        st.markdown("""
        <div class="step-header">
            💡 RACCOMANDAZIONI
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = results["recommendations"]
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Il contenuto è già ben ottimizzato!")
        
        # ========== 8. EXPORT DATA ==========
        st.markdown("---")
        st.subheader("📥 Export Dati")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            import json
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name=f"ai_overview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export delle metriche principali
            summary_data = {
                "URL Target": [target_url],
                "Keyword": [keyword],
                "Relevance Score": [target_analysis.get('relevance_score', 0)],
                "Optimized Score": [optimization['new_relevance_score'] if optimization else 0],
                "Improvement %": [optimization['improvement_percentage'] if optimization else 0],
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
        
    except Exception as e:
        st.error(f"❌ Errore durante l'analisi: {str(e)}")
        st.exception(e)
        logger.error(f"Errore Streamlit: {str(e)}", exc_info=True)

else:
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
    
    ### Requisiti:
    
    - ✅ URL della pagina da ottimizzare
    - ✅ Keyword target
    - ✅ API Keys configurate (DataForSEO, OpenAI, Jina/Google)
    
    ### API Keys Necessarie:
    
    Crea un file `.env` con:
    ```
    DATAFORSEO_LOGIN=your_login
    DATAFORSEO_PASSWORD=your_password
    OPENAI_API_KEY=your_openai_key
    JINA_API_KEY=your_jina_key
    ```
    
    ---
    
    **Inizia inserendo i dati nella sidebar e clicca "AVVIA ANALISI" ⬅️**
    """)
    
    # Info box
    st.info("""
    💡 **Tip**: Per risultati ottimali, scegli keyword informative che già mostrano AI Overview nelle SERP.
    Esempio: "come fare...", "cosa significa...", "guida a..."
    """)
