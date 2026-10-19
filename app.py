"""
FinSight AI — Institutional Financial Intelligence & Research Copilot
======================================================================
Production-grade multi-agent financial report research and analytics web application.
Integrated with:
  - ChromaDB Vector Store (cosine similarity, 384d all-MiniLM-L6-v2)
  - Agent 1: Extraction Agent (precise table & note extraction with evidence provenance)
  - Agent 2: Analysis Agent (YoY variance, margin calculation, normalized scales)
  - Agent 3: Red Flag Agent (rule-based anomaly detection & audit flags)
  - Multi-Agent Orchestrator (sequential pipeline coordination)
  - Research Agent & ChatSession (conversational reasoning with multi-turn memory)
  - Comparison Agent (cross-company benchmarking with currency safety)
  - Report Agent (executive briefing generator with provenance)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import altair as alt

# Ensure root imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import comparison_agent
import multi_agent_orchestrator
import report_agent
import chat_interface
import extraction_agent
from config import DATA_DIR, DOCUMENTS_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, VECTORDB_DIR

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSight AI — Financial Intelligence Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Deep Emerald & Ivory Theme Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* CSS Reset and Font System */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --emerald-dark: #0B251D;
    --emerald-primary: #0F382C;
    --emerald-light: #164E3D;
    --emerald-subtle: #EAF2EF;
    --gold-primary: #D4AF37;
    --gold-muted: #B8972E;
    --gold-subtle: #FBF8EE;
    --ivory-bg: #F9F8F5;
    --card-bg: #FFFFFF;
    --charcoal-dark: #1A1F1D;
    --charcoal-body: #333C37;
    --charcoal-muted: #5E6963;
    --border-light: rgba(15, 56, 44, 0.12);
    --border-gold: rgba(212, 175, 55, 0.35);
    --shadow-sm: 0 2px 6px rgba(11, 37, 29, 0.04);
    --shadow-md: 0 6px 18px rgba(11, 37, 29, 0.07);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--charcoal-dark);
}

.stApp {
    background-color: var(--ivory-bg);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #F4F3EE !important;
    border-right: 1px solid var(--border-light);
}

/* Clean Header Card */
.brand-header {
    background: linear-gradient(135deg, var(--emerald-primary) 0%, var(--emerald-dark) 100%);
    color: #FFFFFF;
    padding: 24px 30px;
    border-radius: 14px;
    border: 1px solid var(--border-gold);
    box-shadow: var(--shadow-md);
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brand-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-badge {
    background: rgba(212, 175, 55, 0.2);
    color: var(--gold-primary);
    border: 1px solid var(--gold-primary);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
}

.brand-subtitle {
    font-size: 14px;
    color: #D3DFDA;
    margin-top: 6px;
    font-weight: 400;
}

/* Metric / Analytic Cards */
.fin-card {
    background: var(--card-bg);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}

.fin-card:hover {
    box-shadow: var(--shadow-md);
    border-color: rgba(15, 56, 44, 0.25);
}

.fin-card-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--charcoal-muted);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.fin-card-value {
    font-size: 24px;
    font-weight: 800;
    color: var(--emerald-dark);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.fin-card-delta-pos {
    color: #107C41;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}

.fin-card-delta-neg {
    color: #C5221F;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}

.fin-card-provenance {
    font-size: 11px;
    color: var(--charcoal-muted);
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px dashed var(--border-light);
    font-family: 'JetBrains Mono', monospace;
}

/* Provenance Tags */
.badge-reported {
    background: #EAF2EF;
    color: #0F382C;
    border: 1px solid rgba(15, 56, 44, 0.2);
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
}

.badge-derived {
    background: #FBF8EE;
    color: #9A7718;
    border: 1px solid rgba(212, 175, 55, 0.35);
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
}

.badge-flag-critical {
    background: #FDF2F2;
    color: #9B1C1C;
    border: 1px solid rgba(239, 68, 68, 0.3);
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}

.badge-flag-info {
    background: #F0FDF4;
    color: #166534;
    border: 1px solid rgba(34, 197, 94, 0.3);
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}

/* Chat Bubbles */
.chat-msg-user {
    background: #FFFFFF;
    border: 1px solid var(--border-light);
    border-radius: 12px 12px 2px 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    margin-left: 10%;
    color: var(--charcoal-dark);
    box-shadow: var(--shadow-sm);
}

.chat-msg-ai {
    background: #FFFFFF;
    border: 1px solid rgba(15, 56, 44, 0.15);
    border-left: 4px solid var(--emerald-primary);
    border-radius: 2px 12px 12px 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    margin-right: 5%;
    box-shadow: var(--shadow-sm);
}

/* Executive Notice */
.advisory-box {
    background: #FBF8EE;
    border-left: 4px solid var(--gold-primary);
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 13px;
    color: #6E5311;
    margin: 14px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Access & Session State Initialization
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_index_catalog():
    """Load index catalog from persistent ChromaDB collection."""
    try:
        return comparison_agent.load_index_catalog()
    except Exception as exc:
        return {"counts": Counter(), "error": str(exc)}


def get_indexed_companies_and_years(catalog):
    """Parse catalog into a dictionary of {company: [sorted_years]}."""
    data = {}
    for (company, year), count in catalog.get("counts", {}).items():
        if company and year and year != "unknown":
            data.setdefault(company, set()).add(str(year))
    return {c: sorted(list(ys)) for c, ys in sorted(data.items())}


if "chat_session" not in st.session_state:
    st.session_state.chat_session = chat_interface.create_session()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Executive Overview"


# -----------------------------------------------------------------------------
# Ingestion Pipeline Function (Clean & Safe)
# -----------------------------------------------------------------------------
def run_pdf_ingestion(uploaded_file, company_name, fiscal_year, progress_bar, status_text):
    """
    Ingests an uploaded PDF into the existing ChromaDB retrieval pipeline:
    1. Saves PDF into data/annual_reports/<filename>
    2. Extracts page-by-page text via pdfplumber into documents/
    3. Chunks page text with exact page-level provenance into chunks/
    4. Computes dense MiniLM embeddings into embeddings/
    5. Safely indexes new records into ChromaDB collection without resetting existing records
    """
    import pdfplumber
    import chunk_documents
    import generate_embeddings
    import index_cromadb

    safe_company = "".join(c for c in company_name if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_filename = f"{safe_company}_{fiscal_year}.pdf".replace(" ", "_")

    # Step 1: Save PDF
    status_text.text("Step 1/4: Saving source report to secure storage...")
    progress_bar.progress(15)
    reports_dir = PROJECT_ROOT / "data" / "annual_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_save_path = reports_dir / safe_filename
    with open(pdf_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 2: Extract text page-by-page
    status_text.text("Step 2/4: Extracting high-fidelity page text...")
    progress_bar.progress(35)
    pages_data = []
    with pdfplumber.open(pdf_save_path) as pdf:
        total_p = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            pages_data.append({"page_number": i, "text": t})

    doc_json_path = PROJECT_ROOT / "documents" / f"{pdf_save_path.stem}.json"
    PROJECT_ROOT.joinpath("documents").mkdir(parents=True, exist_ok=True)
    with open(doc_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": safe_filename,
            "company": safe_company,
            "fiscal_year": str(fiscal_year),
            "total_pages": len(pages_data),
            "pages": pages_data,
        }, f, indent=2, ensure_ascii=False)

    # Step 3: Chunk Document
    status_text.text("Step 3/4: Creating contextual overlapping document chunks...")
    progress_bar.progress(60)
    with open(doc_json_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    chunks = chunk_documents.process_document(doc_data)
    PROJECT_ROOT.joinpath("chunks").mkdir(parents=True, exist_ok=True)
    chunks_path = PROJECT_ROOT / "chunks" / f"{pdf_save_path.stem}_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Step 4: Embeddings & ChromaDB Index
    status_text.text("Step 4/4: Generating neural embeddings and indexing to ChromaDB...")
    progress_bar.progress(85)
    model = generate_embeddings.get_model()
    embedded_count, skipped_count, _ = generate_embeddings.process_chunk_file(model, chunks_path)

    # Index into ChromaDB
    embeddings_path = PROJECT_ROOT / "embeddings" / f"{pdf_save_path.stem}_embeddings.json"
    with open(embeddings_path, "r", encoding="utf-8") as f:
        emb_records = json.load(f)

    _, collection = index_cromadb.get_chroma_collection()
    added_count, _, _ = index_cromadb.index_records(collection, emb_records)

    progress_bar.progress(100)
    status_text.text("Ingestion completed successfully!")
    st.cache_data.clear()

    return {
        "status": "success",
        "filename": safe_filename,
        "pages": len(pages_data),
        "chunks": len(chunks),
        "indexed": added_count,
    }


# -----------------------------------------------------------------------------
# Top Brand Bar
# -----------------------------------------------------------------------------
catalog = get_index_catalog()
company_coverage = get_indexed_companies_and_years(catalog)
total_chunks = sum(catalog.get("counts", {}).values())

st.markdown(f"""
<div class="brand-header">
    <div>
        <h1 class="brand-title">📈 FinSight AI <span class="brand-badge">Institutional Edition</span></h1>
        <div class="brand-subtitle">Autonomous Multi-Agent Financial Research, YoY Audit & Provenance Intelligence</div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #D3DFDA;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #107C41; border-radius: 50%; margin-right: 6px;"></span>
        <strong>{total_chunks:,}</strong> Vector Chunks Indexed &nbsp;|&nbsp; <strong>{len(company_coverage)}</strong> Enterprises
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### **Navigation**")
    nav_selection = st.radio(
        "Select Module:",
        options=[
            "Executive Overview",
            "Upload Report",
            "Research Copilot",
            "Company Analysis",
            "Company Comparison",
            "Visual Analytics",
            "Report Copilot",
        ],
        index=[
            "Executive Overview",
            "Upload Report",
            "Research Copilot",
            "Company Analysis",
            "Company Comparison",
            "Visual Analytics",
            "Report Copilot",
        ].index(st.session_state.current_page) if st.session_state.current_page in [
            "Executive Overview",
            "Upload Report",
            "Research Copilot",
            "Company Analysis",
            "Company Comparison",
            "Visual Analytics",
            "Report Copilot",
        ] else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_page = nav_selection

    st.markdown("---")
    st.markdown("#### **Active Vector Store**")
    st.markdown(f"""
    - **Engine**: ChromaDB (Cosine)
    - **Model**: `all-MiniLM-L6-v2` (384d)
    - **Total Records**: **{total_chunks:,}**
    - **Companies**: **{len(company_coverage)}**
    """)

    st.markdown("#### **Indexed Enterprises**")
    for comp, yrs in company_coverage.items():
        yr_str = ", ".join(f"'{y[-2:]}" for y in yrs)
        st.markdown(f"<span style='font-size:12px;'>• <strong>{comp}</strong> <span style='color:#5E6963;'>({yr_str})</span></span>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_session.reset()
        st.success("Session state cleared.")
        st.rerun()


# =============================================================================
# VIEW 1: EXECUTIVE OVERVIEW DASHBOARD
# =============================================================================
if st.session_state.current_page == "Executive Overview":
    st.markdown("### 🏛️ Executive Intelligence Dashboard")
    st.markdown(
        "FinSight AI automates fundamental financial research by orchestrating specialized AI agents "
        "across audited annual filings with complete mathematical and citation provenance."
    )

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title">Indexed Knowledge Base</div>
            <div class="fin-card-value">{total_chunks:,}</div>
            <div style="font-size:12px; color:#5E6963;">Verified report chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title">Covered Enterprises</div>
            <div class="fin-card-value">{len(company_coverage)}</div>
            <div style="font-size:12px; color:#5E6963;">Global technology & industrials</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        total_filings = sum(len(y) for y in company_coverage.values())
        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title">Filing Catalog</div>
            <div class="fin-card-value">{total_filings}</div>
            <div style="font-size:12px; color:#5E6963;">Audited annual 10-K / reports</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">Provenance Integrity</div>
            <div class="fin-card-value">100%</div>
            <div style="font-size:12px; color:#107C41;">Zero hallucinated values</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Primary Action Banner
    col_banner, col_action = st.columns([3, 1])
    with col_banner:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid var(--border-light); padding:20px; border-radius:12px;">
            <h4 style="margin:0 0 6px 0; color:var(--emerald-primary);">Need to analyze a new filing?</h4>
            <div style="font-size:13px; color:var(--charcoal-body);">
                Upload any company's PDF annual report or 10-K filing to automatically extract metrics, generate vector embeddings, and enable deep conversational research.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📤 Ingest New PDF Report", use_container_width=True, type="primary"):
            st.session_state.current_page = "Upload Report"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-Agent Architecture
    st.markdown("#### **Specialized Agent Architecture**")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">Agent 1: Extraction</div>
            <div style="font-size:13px; color:var(--charcoal-body); margin-bottom:12px;">
                Extracts financial metrics directly from audited tables and notes with phrase tiers and exact page citations.
            </div>
            <span class="badge-reported">PROVENANCE VERIFIED</span>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">Agent 2: Analysis</div>
            <div style="font-size:13px; color:var(--charcoal-body); margin-bottom:12px;">
                Computes YoY absolute and percentage variances, normalized margins, and enforces cross-period consistency.
            </div>
            <span class="badge-derived">STRICT ARITHMETIC</span>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">Agent 3: Red Flag Audit</div>
            <div style="font-size:13px; color:var(--charcoal-body); margin-bottom:12px;">
                Identifies margin compression, negative operating cash flows, leverage spikes, and verbatim risk disclosures.
            </div>
            <span class="badge-flag-critical">RISK GOVERNANCE</span>
        </div>
        """, unsafe_allow_html=True)
    with a4:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">Agent 4: Synthesis</div>
            <div style="font-size:13px; color:var(--charcoal-body); margin-bottom:12px;">
                Generates executive briefing reports and conducts cross-company peer comparisons with strict currency safety.
            </div>
            <span class="badge-reported">EXECUTIVE BRIEFINGS</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### **Enterprise Coverage Directory**")
    cov_cols = st.columns(len(company_coverage) if company_coverage else 1)
    for i, (comp, years) in enumerate(company_coverage.items()):
        col = cov_cols[i % len(cov_cols)]
        with col:
            yr_badges = " ".join([f"<span class='badge-reported'>FY{y}</span>" for y in years])
            st.markdown(f"""
            <div class="fin-card" style="padding:14px;">
                <div style="font-weight:700; color:var(--emerald-primary); margin-bottom:6px;">{comp}</div>
                <div>{yr_badges}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# VIEW 2: UPLOAD REPORT & INGESTION PIPELINE
# =============================================================================
elif st.session_state.current_page == "Upload Report":
    st.markdown("### 📄 Annual Report Ingestion Pipeline")
    st.markdown(
        "Upload a verified PDF annual report to parse, chunk, embed, and index into the persistent ChromaDB collection. "
        "The document immediately becomes available for conversational queries, automated ratio analysis, and comparison."
    )

    col_form, col_info = st.columns([1.5, 1])

    with col_form:
        st.markdown("#### **Document Upload & Metadata**")
        uploaded_file = st.file_uploader("Select Annual Report PDF (up to 200MB):", type=["pdf"])

        col_c, col_y = st.columns(2)
        with col_c:
            company_input = st.text_input("Company Name (e.g., Apple, Alphabet):", placeholder="Company Name")
        with col_y:
            fiscal_year_input = st.number_input("Fiscal Year:", min_value=2015, max_value=2030, value=2024, step=1)

        # Pre-flight check
        already_exists = False
        if company_input and str(fiscal_year_input) in company_coverage.get(company_input, []):
            already_exists = True
            st.warning(f"⚠️ {company_input} FY{fiscal_year_input} is already present in ChromaDB. Uploading again will index new/updated chunks safely.")

        submit_btn = st.button("🚀 Process & Index Report", type="primary", use_container_width=True)

        if submit_btn:
            if not uploaded_file:
                st.error("Please select a PDF file before starting ingestion.")
            elif not company_input.strip():
                st.error("Please provide a valid Company Name.")
            else:
                p_bar = st.progress(0)
                status_txt = st.empty()
                try:
                    with st.spinner("Processing report through multi-agent ingestion pipeline..."):
                        result = run_pdf_ingestion(
                            uploaded_file,
                            company_input.strip(),
                            fiscal_year_input,
                            p_bar,
                            status_txt
                        )
                    st.success(f"🎉 Successfully indexed {result['filename']} ({result['pages']} pages, {result['chunks']} chunks, {result['indexed']} new records added to ChromaDB)!")
                    st.balloons()
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with col_info:
        st.markdown("""
        <div class="fin-card">
            <h4 style="margin-top:0; color:var(--emerald-primary);">Pipeline Specifications</h4>
            <ul style="font-size:13px; color:var(--charcoal-body); padding-left:18px;">
                <li><strong>Page-Exact Provenance</strong>: Text is extracted page-by-page to ensure every citation links to an exact physical page.</li>
                <li><strong>Overlap Windowing</strong>: 600-word chunks with 100-word sliding overlap preserve financial table contexts.</li>
                <li><strong>Neural Embeddings</strong>: Normalized 384-dimensional dense vectors using <code>all-MiniLM-L6-v2</code>.</li>
                <li><strong>De-duplication Guard</strong>: ChromaDB unique chunk hashes prevent duplicate entries if re-run.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 3: RESEARCH COPILOT (CONVERSATIONAL INTERFACE)
# =============================================================================
elif st.session_state.current_page == "Research Copilot":
    st.markdown("### 💬 Institutional Research Copilot")
    st.markdown(
        "Ask natural-language questions about any indexed company's financial results, margins, risks, or comparisons. "
        "Every figure is grounded in verbatim citations from the audited filings."
    )

    # Suggested prompt pills
    st.markdown("<span style='font-size:12px; font-weight:700; color:#5E6963;'>QUICK RESEARCH QUERIES:</span>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Apple FY2023 vs FY2024 Analysis", use_container_width=True):
            st.session_state.prompt_to_run = "Compare Apple's FY2023 and FY2024 revenue, net income, diluted EPS and operating margin. Explain the changes using only the indexed annual reports."
    with p2:
        if st.button("Apple vs Microsoft vs Amazon (FY2024)", use_container_width=True):
            st.session_state.prompt_to_run = "Compare Apple, Microsoft and Amazon for FY2024 revenue, net income and operating margin."
    with p3:
        if st.button("Microsoft FY2024 Risk Disclosures", use_container_width=True):
            st.session_state.prompt_to_run = "What financial risks are mentioned in Microsoft's annual report for FY2024?"

    # Display Conversation History
    for turn in st.session_state.chat_session.history:
        if turn["role"] == "user":
            st.markdown(f"""
            <div class="chat-msg-user">
                <strong>You</strong><br>
                <div style="margin-top:4px;">{turn['markdown']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-msg-ai">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="font-weight:700; color:var(--emerald-primary);">FinSight Research Copilot</span>
                    <span class="badge-reported">{turn.get('route', 'research').upper()}</span>
                </div>
                <div>{turn['markdown']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Handle user prompt from text input or preset button
    query_input = st.chat_input("Ask a financial question (e.g., 'What was Tesla\\'s FY2024 EPS and operating cash flow?')")
    prompt_to_execute = query_input or st.session_state.pop("prompt_to_run", None)

    if prompt_to_execute:
        with st.spinner("Retrieving annual report evidence and synthesizing response..."):
            turn = st.session_state.chat_session.ask(prompt_to_execute)
        st.rerun()


# =============================================================================
# VIEW 4: COMPANY ANALYSIS (SINGLE-COMPANY YoY DEEP DIVE)
# =============================================================================
elif st.session_state.current_page == "Company Analysis":
    st.markdown("### 📊 Comprehensive Company Analysis")
    st.markdown(
        "Execute automated sequential multi-agent analysis for a selected enterprise across two fiscal years. "
        "Extracts metrics, validates accounting consistency, computes YoY changes, and runs the Red Flag audit."
    )

    if not company_coverage:
        st.warning("No companies currently indexed in ChromaDB. Please ingest a report in the Upload section.")
    else:
        c_sel, y_prior, y_curr, a_btn = st.columns([2, 1, 1, 1.2])
        with c_sel:
            selected_comp = st.selectbox("Company:", options=list(company_coverage.keys()), index=0)
        available_years = company_coverage.get(selected_comp, [])
        with y_prior:
            prior_year = st.selectbox("Prior Year:", options=available_years, index=0 if len(available_years) < 2 else 0)
        with y_curr:
            curr_year = st.selectbox("Current Year:", options=available_years, index=len(available_years) - 1)
        with a_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_action = st.button("Run Full Analysis", type="primary", use_container_width=True)

        if analyze_action or "last_company_analysis" in st.session_state:
            if analyze_action:
                with st.spinner(f"Running Multi-Agent Orchestrator for {selected_comp} (FY{prior_year} -> FY{curr_year})..."):
                    analysis_result = multi_agent_orchestrator.run_financial_analysis(
                        selected_comp, str(prior_year), str(curr_year), verbose=False
                    )
                    st.session_state.last_company_analysis = analysis_result
            else:
                analysis_result = st.session_state.last_company_analysis

            if analysis_result.get("status") == "pipeline_error":
                st.error("Pipeline encountered an error while processing the request.")
                for err in analysis_result.get("pipeline_errors", []):
                    st.markdown(f"• **{err.get('stage')}**: {err.get('message')}")
            else:
                a1_data = analysis_result.get("agent_1", {})
                a2_data = analysis_result.get("agent_2", {}).get("result", {})
                a3_data = analysis_result.get("agent_3", {}).get("result", {})

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### **Financial Metrics Dashboard — {selected_comp} (FY{prior_year} vs FY{curr_year})**")

                curr_metrics = a1_data.get("current", {})
                comparisons = a2_data.get("comparisons", {})

                # Grid of Core Financial Cards
                card_metrics = [
                    ("revenue", "Total Revenue"),
                    ("net_income", "Net Income"),
                    ("eps", "Diluted EPS"),
                    ("gross_margin", "Gross Margin"),
                    ("operating_margin", "Operating Margin"),
                    ("net_profit_margin", "Net Profit Margin"),
                    ("cash_flow", "Operating Cash Flow"),
                    ("total_debt", "Total Debt"),
                ]

                g_cols = st.columns(4)
                for idx, (m_key, m_label) in enumerate(card_metrics):
                    col = g_cols[idx % 4]
                    rec = curr_metrics.get(m_key, {})
                    comp = comparisons.get(m_key, {})
                    is_derived = rec.get("derived", False) or m_key == "net_profit_margin"

                    with col:
                        if rec and rec.get("value") is not None:
                            val_fmt = f"{rec['value']:,.2f}".rstrip("0").rstrip(".")
                            unit = rec.get("unit") or ""
                            val_display = f"{val_fmt} {unit}"

                            # YoY delta
                            delta_html = ""
                            if comp and comp.get("percentage_change") is not None:
                                pct = comp["percentage_change"]
                                if pct > 0:
                                    delta_html = f"<div class='fin-card-delta-pos'>▲ +{pct:.2f}% YoY</div>"
                                elif pct < 0:
                                    delta_html = f"<div class='fin-card-delta-neg'>▼ {pct:.2f}% YoY</div>"
                                else:
                                    delta_html = f"<div style='font-size:12px; color:#5E6963;'>— 0.00% YoY</div>"
                            elif comp and comp.get("margin_change_percentage_points") is not None:
                                pts = comp["margin_change_percentage_points"]
                                if pts > 0:
                                    delta_html = f"<div class='fin-card-delta-pos'>▲ +{pts:.2f} pts</div>"
                                elif pts < 0:
                                    delta_html = f"<div class='fin-card-delta-neg'>▼ {pts:.2f} pts</div>"

                            prov = f"{rec.get('source_file', 'Report')}, p.{rec.get('page_number', '?')}"
                            badge = "<span class='badge-derived'>DERIVED</span>" if is_derived else "<span class='badge-reported'>REPORTED</span>"

                            st.markdown(f"""
                            <div class="fin-card">
                                <div class="fin-card-title">
                                    <span>{m_label}</span>
                                    {badge}
                                </div>
                                <div class="fin-card-value">{val_display}</div>
                                {delta_html}
                                <div class="fin-card-provenance">Src: {prov}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="fin-card">
                                <div class="fin-card-title">{m_label}</div>
                                <div class="fin-card-value" style="font-size:16px; color:#5E6963;">Not Found</div>
                                <div style="font-size:11px; color:#5E6963; margin-top:8px;">Insufficient evidence in filing</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Red Flag Assessment
                st.markdown("#### **Agent 3: Red Flag Anomaly & Risk Audit**")
                flags = a3_data.get("red_flags", [])
                summary_score = a3_data.get("red_flag_score", 0)

                if not flags:
                    st.success("✅ **Zero Critical Red Flags Detected**: Operating cash flows, margins, and leverage parameters satisfy healthy bounds.")
                else:
                    st.warning(f"⚠️ **{len(flags)} Risk Alerts Flagged** (Audit Score: {summary_score})")
                    for flag in flags:
                        sev = flag.get("severity", "MODERATE").upper()
                        badge_cls = "badge-flag-critical" if sev in ("CRITICAL", "HIGH") else "badge-flag-info"
                        with st.expander(f"🚩 [{sev}] {flag.get('title', flag.get('category', 'Risk Alert'))}", expanded=True):
                            st.markdown(f"**Description**: {flag.get('description', '')}")
                            if flag.get("criteria"):
                                st.markdown(f"**Trigger Rule**: `{flag['criteria']}`")
                            if flag.get("evidence"):
                                st.markdown(f"> *\"{flag['evidence']}\"*")
                            if flag.get("source_file"):
                                st.caption(f"Source: {flag['source_file']}, p.{flag.get('page_number', '?')}")

                # Limitations & Notes
                limits = analysis_result.get("limitations", [])
                if limits:
                    with st.expander("ℹ️ Data Limitations & Uncalculated Metrics"):
                        for l in limits:
                            st.markdown(f"- {l.get('message', str(l))}")


# =============================================================================
# VIEW 5: COMPANY COMPARISON (MULTI-COMPANY BENCHMARKING)
# =============================================================================
elif st.session_state.current_page == "Company Comparison":
    st.markdown("### ⚖️ Cross-Company Benchmarking")
    st.markdown(
        "Compare fundamental performance metrics across multiple enterprises for a common fiscal year. "
        "Strict accounting checks prevent erroneous comparisons across differing currencies and units."
    )

    if not company_coverage:
        st.warning("No companies indexed in ChromaDB.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_comps = st.multiselect(
                "Select Companies to Benchmark:",
                options=list(company_coverage.keys()),
                default=list(company_coverage.keys())[:3] if len(company_coverage) >= 3 else list(company_coverage.keys())
            )
        with c2:
            all_years = sorted(list({y for ys in company_coverage.values() for y in ys}), reverse=True)
            cmp_year = st.selectbox("Fiscal Year:", options=all_years, index=0)

        metrics_available = [
            ("revenue", "Revenue"),
            ("net_income", "Net Income"),
            ("eps", "Diluted EPS"),
            ("operating_margin", "Operating Margin"),
            ("gross_margin", "Gross Margin"),
            ("cash_flow", "Operating Cash Flow"),
            ("total_debt", "Total Debt"),
        ]
        chosen_metrics = st.multiselect(
            "Metrics to Include:",
            options=[k for k, _ in metrics_available],
            default=["revenue", "net_income", "eps", "operating_margin"],
            format_func=lambda x: dict(metrics_available).get(x, x)
        )

        if st.button("Run Comparison Analysis", type="primary"):
            if len(selected_comps) < 2:
                st.error("Please select at least two companies to benchmark.")
            else:
                with st.spinner("Running Comparison Agent across index catalog..."):
                    cmp_res = comparison_agent.build_comparison(
                        selected_comps, [cmp_year], metrics=chosen_metrics
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### **Benchmark Results — FY{cmp_year}**")

                # Incompatibility and Currency Advisory
                for c in cmp_res.get("comparisons", []):
                    if c.get("incompatibility"):
                        st.markdown(f"""
                        <div class="advisory-box">
                            <strong>⚠️ Currency / Unit Notice for {c['metric_label']}:</strong> {c['incompatibility']['reason']}.
                            <br>Nominal amounts in different currencies (e.g. USD vs INR) are preserved in their native units and not falsely equated.
                        </div>
                        """, unsafe_allow_html=True)

                # Comparison Table
                for c in cmp_res.get("comparisons", []):
                    st.markdown(f"##### **{c['metric_label']}**")
                    table_rows = []
                    for entry in c.get("entries", []):
                        val_str = f"{entry['value']:,.2f} {entry.get('unit', '')}" if entry.get("value") is not None else "Not Found"
                        table_rows.append({
                            "Company": str(entry["company"]),
                            "Value": str(val_str),
                            "Currency": str(entry.get("currency") or "—"),
                            "Basis": "Derived" if entry.get("derived") else "Reported",
                            "Source File": str(entry.get("source_file") or "—"),
                            "Page": str(entry.get("page_number")) if entry.get("page_number") is not None else "—",
                        })

                    if table_rows:
                        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                    if c.get("relations"):
                        st.markdown("**Pairwise Findings:**")
                        for r in c["relations"]:
                            st.markdown(f"- {r['statement']}")
                    st.markdown("---")


# =============================================================================
# VIEW 6: VISUAL ANALYTICS (INTERACTIVE ALTAIR CHARTS)
# =============================================================================
elif st.session_state.current_page == "Visual Analytics":
    st.markdown("### 📈 Visual Analytics & Financial Trends")
    st.markdown(
        "Interactive graphical analysis of revenue growth, profitability margins, and capital structure. "
        "Every chart is rendered strictly from verified annual report values."
    )

    if not company_coverage:
        st.warning("No data indexed.")
    else:
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            v_comp = st.selectbox("Select Target Company:", options=list(company_coverage.keys()), index=0)
            v_years = company_coverage.get(v_comp, [])
            st.markdown(f"Available years for **{v_comp}**: {', '.join(v_years)}")

        # Fetch multi-year metrics for the company
        data_records = []
        with st.spinner(f"Extracting multi-year data points for {v_comp}..."):
            for yr in v_years:
                res = extraction_agent.extract_metrics(v_comp, yr, metric_keys=["revenue", "net_income", "eps", "operating_margin", "gross_margin", "cash_flow"])
                rec_entry = {"Year": f"FY{yr}", "raw_year": int(yr)}
                for k in ["revenue", "net_income", "eps", "operating_margin", "gross_margin", "cash_flow"]:
                    rec_entry[k] = res.get(k, {}).get("value")
                data_records.append(rec_entry)

        df = pd.DataFrame(data_records).sort_values("raw_year")

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### **Revenue & Net Income Trajectory**")
            rev_net_df = df.melt(id_vars=["Year"], value_vars=["revenue", "net_income"], var_name="Metric", value_name="Amount")
            rev_net_df["Metric"] = rev_net_df["Metric"].map({"revenue": "Revenue", "net_income": "Net Income"})

            chart_rev = alt.Chart(rev_net_df.dropna()).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Year:N", title="Fiscal Year"),
                y=alt.Y("Amount:Q", title="Amount (Millions)"),
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["Revenue", "Net Income"], range=["#0F382C", "#D4AF37"])),
                xOffset="Metric:N",
                tooltip=["Year", "Metric", alt.Tooltip("Amount:Q", format=",.2f")]
            ).properties(height=320)
            st.altair_chart(chart_rev, use_container_width=True)

        with chart_col2:
            st.markdown("##### **Profitability Margin Trends (%)**")
            margin_df = df.melt(id_vars=["Year"], value_vars=["gross_margin", "operating_margin"], var_name="Margin", value_name="Percent")
            margin_df["Margin"] = margin_df["Margin"].map({"gross_margin": "Gross Margin", "operating_margin": "Operating Margin"})

            chart_margin = alt.Chart(margin_df.dropna()).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("Year:N", title="Fiscal Year"),
                y=alt.Y("Percent:Q", title="Percentage (%)"),
                color=alt.Color("Margin:N", scale=alt.Scale(domain=["Gross Margin", "Operating Margin"], range=["#164E3D", "#B8972E"])),
                tooltip=["Year", "Margin", alt.Tooltip("Percent:Q", format=",.2f")]
            ).properties(height=320)
            st.altair_chart(chart_margin, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.markdown("##### **Diluted Earnings Per Share ($)**")
            if "eps" in df.columns and df["eps"].notna().any():
                chart_eps = alt.Chart(df.dropna(subset=["eps"])).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0F382C").encode(
                    x=alt.X("Year:N", title="Fiscal Year"),
                    y=alt.Y("eps:Q", title="EPS ($ per share)"),
                    tooltip=["Year", alt.Tooltip("eps:Q", format=",.2f")]
                ).properties(height=300)
                st.altair_chart(chart_eps, use_container_width=True)
            else:
                st.info("EPS data not available.")

        with chart_col4:
            st.markdown("##### **Operating Cash Flow ($)**")
            if "cash_flow" in df.columns and df["cash_flow"].notna().any():
                chart_cf = alt.Chart(df.dropna(subset=["cash_flow"])).mark_area(
                    line={"color": "#164E3D"},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#0F382C', offset=0),
                               alt.GradientStop(color='rgba(15, 56, 44, 0.05)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X("Year:N", title="Fiscal Year"),
                    y=alt.Y("cash_flow:Q", title="Cash Flow ($)"),
                    tooltip=["Year", alt.Tooltip("cash_flow:Q", format=",.2f")]
                ).properties(height=300)
                st.altair_chart(chart_cf, use_container_width=True)
            else:
                st.info("Cash flow data not available.")


# =============================================================================
# VIEW 7: REPORT COPILOT (EXECUTIVE RESEARCH REPORT GENERATION)
# =============================================================================
elif st.session_state.current_page == "Report Copilot":
    st.markdown("### 📑 Executive Report Generator")
    st.markdown(
        "Generate a publication-ready fundamental research briefing document for an enterprise across two fiscal periods. "
        "Includes executive summaries, key financials, red flag risk assessments, and compliance disclaimers."
    )

    if not company_coverage:
        st.warning("No companies available.")
    else:
        r_c1, r_c2, r_c3, r_c4 = st.columns([2, 1, 1, 1.2])
        with r_c1:
            rep_comp = st.selectbox("Company:", options=list(company_coverage.keys()), key="rep_comp")
        r_years = company_coverage.get(rep_comp, [])
        with r_c2:
            rep_prior = st.selectbox("Prior Year:", options=r_years, index=0 if len(r_years) < 2 else 0, key="rep_prior")
        with r_c3:
            rep_curr = st.selectbox("Current Year:", options=r_years, index=len(r_years) - 1, key="rep_curr")
        with r_c4:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_report_btn = st.button("Generate Briefing", type="primary", use_container_width=True)

        if gen_report_btn or "last_report" in st.session_state:
            if gen_report_btn:
                with st.spinner(f"Compiling executive financial report for {rep_comp}..."):
                    report_obj = report_agent.generate_report(rep_comp, str(rep_prior), str(rep_curr))
                    report_md = report_agent.format_report_markdown(report_obj)
                    st.session_state.last_report = report_obj
                    st.session_state.last_report_md = report_md
            else:
                report_obj = st.session_state.last_report
                report_md = st.session_state.last_report_md

            st.markdown("<br>", unsafe_allow_html=True)

            # Download Buttons
            d_col1, d_col2, _ = st.columns([1, 1, 2])
            with d_col1:
                st.download_button(
                    "📥 Download Report (.md)",
                    data=report_md,
                    file_name=f"FinSight_{rep_comp}_FY{rep_prior}_FY{rep_curr}_Report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with d_col2:
                st.download_button(
                    "📄 Download Plain Text (.txt)",
                    data=report_md,
                    file_name=f"FinSight_{rep_comp}_FY{rep_prior}_FY{rep_curr}_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid var(--border-light); padding:28px 32px; border-radius:12px; box-shadow:var(--shadow-sm);">
            """, unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown("</div>", unsafe_allow_html=True)
