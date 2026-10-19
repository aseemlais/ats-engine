import os
import sys
from pathlib import Path

# Fix SSL certificate path on Windows environments if missing/stale
try:
    import certifi
    ca_file = certifi.where()
    if os.path.exists(ca_file):
        os.environ["SSL_CERT_FILE"] = ca_file
        os.environ["REQUESTS_CA_BUNDLE"] = ca_file
        os.environ["CURL_CA_BUNDLE"] = ca_file
except Exception:
    pass

import streamlit as st
import time
import pandas as pd
# Ensure app modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.database.storage import MetadataStorage
from app.services.jd_parser import JDParserService
from app.services.hybrid_ranker import HybridRankerService
from app.services.embedding_service import EmbeddingService
from app.workers.ingestion_pipeline import IngestionPipeline

st.set_page_config(
    page_title="Scalable Semantic ATS",
    page_icon="🎯",
    layout="wide"
)

# Initialize services
@st.cache_resource
def get_services():
    storage = MetadataStorage()
    jd_parser = JDParserService()
    ranker = HybridRankerService()
    embedding_service = EmbeddingService()
    pipeline = IngestionPipeline()
    return storage, jd_parser, ranker, embedding_service, pipeline

storage, jd_parser, ranker, embedding_service, pipeline = get_services()

st.title("🎯 Scalable Hybrid Semantic ATS (10,000+ Resumes)")
st.caption("Deterministic Information Retrieval | BM25 + Sentence Transformers + Skill Matching + Certifications + Explainable Ranking")

# Sidebar: Stats & Weights
with st.sidebar:
    st.header("📊 System Status")
    resume_count = storage.count_resumes()
    st.metric("Total Indexed Resumes", f"{resume_count:,}")
    
    st.divider()
    st.header("⚙️ Scoring Weights")
    w_sem = st.slider("Semantic Similarity", 0.0, 1.0, config.semantic_weight, 0.05)
    w_bm25 = st.slider("BM25 Lexical", 0.0, 1.0, config.bm25_weight, 0.05)
    w_skill = st.slider("Exact Skill Overlap", 0.0, 1.0, config.skill_weight, 0.05)
    w_exp = st.slider("Experience Match", 0.0, 1.0, config.experience_weight, 0.05)
    w_cert = st.slider("Certifications Match", 0.0, 1.0, config.certification_weight, 0.02)
    
    tot_weight = w_sem + w_bm25 + w_skill + w_exp + w_cert
    if abs(tot_weight - 1.0) > 1e-4:
        st.warning(f"Weights sum to {tot_weight:.2f} (normalized internally)")

    st.divider()
    st.header("🛠️ Hard Filters")
    enable_hard_filter = st.checkbox("Enable Hard Pre-Filter", value=config.hard_filter_enabled)
    strict_mode = st.checkbox("Strict Mandatory Skills & Exp", value=False)
    top_k = st.slider("Top-K Candidates", 10, 500, 50, 10)

# Main tabs
tab_search, tab_upload, tab_benchmarks = st.tabs(["🔍 Candidate Search & Ranking", "📁 Ingest Resumes", "📈 System Benchmark"])

with tab_search:
    st.subheader("1. Enter or Upload Job Description")
    
    col_jd_mode1, col_jd_mode2 = st.columns([1, 1])
    with col_jd_mode1:
        sample_jds = {
            "Custom Job Description": "",
            "Senior NLP / Machine Learning Engineer": """Job Title: Senior NLP Engineer
Requirements:
- 3+ years of professional experience in NLP and Machine Learning
- Strong proficiency in Python, PyTorch, Transformers, HuggingFace
- Experience with text classification, NER, and information retrieval
- Bachelor's or Master's degree in Computer Science or related field
Preferred Skills:
- Docker, FastAPI, Kubernetes, AWS
- Experience with vector search (Qdrant, Faiss) and BM25
Responsibilities:
- Build high-scale resume retrieval and semantic ranking pipelines.""",
            "Senior Full Stack Python Developer": """Job Title: Senior Full Stack Developer
Requirements:
- 4+ years software development experience
- Strong Python, FastAPI, Django, PostgreSQL, Docker
- React, JavaScript, TypeScript frontend experience
- Bachelor's degree in Engineering
Preferred Skills:
- Redis, AWS, Kubernetes, CI/CD pipelines
Responsibilities:
- Design and scale distributed backend APIs and responsive web interfaces."""
        }
        jd_selection = st.selectbox("Choose a sample JD or start blank:", list(sample_jds.keys()))

    with col_jd_mode2:
        jd_file = st.file_uploader("📎 Or Upload JD File (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

    uploaded_jd_text = ""
    if jd_file is not None:
        try:
            if jd_file.name.lower().endswith(".pdf"):
                import fitz
                pdf_doc = fitz.open(stream=jd_file.read(), filetype="pdf")
                extracted_pages = [page.get_text() for page in pdf_doc]
                uploaded_jd_text = "\n".join(extracted_pages).strip()
            elif jd_file.name.lower().endswith(".docx"):
                from docx import Document
                import io
                docx_doc = Document(io.BytesIO(jd_file.read()))
                uploaded_jd_text = "\n".join([p.text for p in docx_doc.paragraphs if p.text.strip()]).strip()
            elif jd_file.name.lower().endswith(".txt"):
                uploaded_jd_text = jd_file.read().decode("utf-8", errors="replace").strip()
            
            if uploaded_jd_text:
                st.info(f"Loaded text from uploaded JD file: **{jd_file.name}**")
        except Exception as e:
            st.error(f"Error reading JD file: {str(e)}")

    initial_jd_val = uploaded_jd_text if uploaded_jd_text else sample_jds[jd_selection]
    jd_text = st.text_area("Job Description Content (Paste or Edit)", value=initial_jd_val, height=200)

    if st.button("🚀 Search & Rank Shortlist", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Please provide a Job Description by pasting text or uploading a file.")
        elif resume_count == 0:
            st.error("No resumes indexed yet. Please inject resumes using the 'Ingest Resumes' tab.")
        else:
            with st.spinner("Executing multi-stage filtering and hybrid ranking..."):
                t_start = time.time()
                
                # Parse JD
                parsed_jd = jd_parser.parse_job_description(jd_text, job_id="JD_SEARCH")
                
                # Fetch all resumes
                all_resumes = storage.get_all_resumes()
                
                # Compute/retrieve vectors for candidates
                vec_map = {}
                for r in all_resumes:
                    vec_map[r.candidate_id] = embedding_service.compute_resume_vectors(r)

                # Configure ranker weights
                weights_dict = {
                    "semantic": w_sem / tot_weight if tot_weight > 0 else 0.40,
                    "bm25": w_bm25 / tot_weight if tot_weight > 0 else 0.27,
                    "skill": w_skill / tot_weight if tot_weight > 0 else 0.15,
                    "experience": w_exp / tot_weight if tot_weight > 0 else 0.10,
                    "certification": w_cert / tot_weight if tot_weight > 0 else 0.08,
                }
                ranker.hard_filter_service.enabled = enable_hard_filter
                ranker.hard_filter_service.strict = strict_mode

                response = ranker.rank_candidates(
                    resumes=all_resumes,
                    resume_vectors_map=vec_map,
                    jd=parsed_jd,
                    top_k=top_k,
                    weights=weights_dict
                )
                
                total_duration = time.time() - t_start

            st.success(f"Search complete in {total_duration * 1000:.1f} ms!")
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Ingested Pool", f"{response.total_indexed_resumes:,}")
            col2.metric("Passed Hard Filter", f"{response.hard_filtered_count:,}")
            col3.metric("Scored & Retrieved", f"{response.retrieved_count:,}")
            col4.metric("Top Ranked Shortlist", f"{response.final_count}")

            st.divider()

            # Results Table
            table_data = []
            for c in response.results:
                table_data.append({
                    "Rank": f"#{c.rank}",
                    "Candidate ID": c.candidate_id,
                    "Name": c.name or "Unknown",
                    "Final Score": f"{c.final_score:.1f}%",
                    "Semantic": f"{c.semantic_score * 100:.1f}%",
                    "BM25": f"{c.bm25_score * 100:.1f}%",
                    "Skills": f"{c.skill_score * 100:.1f}%",
                    "Experience": f"{c.candidate_experience_years:.1f} yrs",
                    "Certs": f"{c.certification_score * 100:.1f}%",
                    "Education": c.education_tier.title(),
                    "Skills Matched": len(c.matched_required_skills),
                    "Skills Missing": len(c.missing_required_skills)
                })

            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True)

            # Detailed Candidate Explanations
            st.subheader("📋 Explainable Candidate Breakdown")
            for c in response.results[:10]:
                with st.expander(f"Rank #{c.rank} - {c.candidate_id} ({c.name}) - Score: {c.final_score:.1f}%"):
                    st.code(c.explanation, language="text")

with tab_upload:
    st.subheader("2. Inject Resumes from File Explorer")
    st.write("Inject your real resumes directly into the ATS without any synthetic generation.")

    inject_mode = st.radio(
        "Choose injection method:",
        ["Option A: Select/Upload Files from File Explorer", "Option B: Enter Local Folder Path directly"],
        horizontal=True
    )

    if inject_mode == "Option A: Select/Upload Files from File Explorer":
        uploaded_files = st.file_uploader(
            "Select resumes from your computer (PDF, DOCX, TXT):",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )
        
        if st.button("📥 Ingest Uploaded Resumes", type="primary"):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                config.resumes_dir.mkdir(parents=True, exist_ok=True)
                for i, f in enumerate(uploaded_files):
                    dest = config.resumes_dir / f.name
                    with open(dest, "wb") as out:
                        out.write(f.getbuffer())
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    status_text.text(f"Saved {f.name}")

                status_text.text("Ingesting, extracting structured attributes, and indexing...")
                res = pipeline.ingest_directory(config.resumes_dir)
                st.success(f"✅ Ingestion Finished! Total in System: {res['total_indexed']}, Newly Added: {res['newly_parsed']}, Duplicates Skipped: {res['duplicates']}")
                st.rerun()
            else:
                st.warning("Please choose one or more resume files first.")

    else:
        folder_path_input = st.text_input(
            "Enter full path to folder containing your resumes on Windows:",
            placeholder=r"e.g. C:\Users\LENOVO\Documents\MyResumes"
        )
        
        col_btn, col_chk = st.columns([1, 1])
        with col_btn:
            start_folder_ingest = st.button("📥 Ingest All Resumes from Folder", type="primary")
        with col_chk:
            reset_chk = st.checkbox("Re-index all files (reset checkpoint)", value=False)
            
        if start_folder_ingest:
            if not folder_path_input.strip():
                st.error("Please specify a valid folder path.")
            else:
                target_folder = Path(folder_path_input.strip().strip('"').strip("'"))
                if not target_folder.exists() or not target_folder.is_dir():
                    st.error(f"Directory not found: `{target_folder}`")
                else:
                    found_files = [f for f in target_folder.glob("*") if f.suffix.lower() in (".pdf", ".docx", ".txt")]
                    if not found_files:
                        st.warning(f"No PDF, DOCX, or TXT files found in `{target_folder}`.")
                    else:
                        with st.spinner(f"Processing {len(found_files)} files from {target_folder.name}..."):
                            res = pipeline.ingest_directory(target_folder, reset_checkpoint=reset_chk)
                        st.success(f"✅ Successfully Ingested from Folder! Total in System: {res['total_indexed']}, Newly Added: {res['newly_parsed']}, Duplicates Skipped: {res['duplicates']}")
                        st.rerun()

with tab_benchmarks:
    st.subheader("3. System Performance & Latency")
    st.write("Check live hardware and indexing statistics:")
    st.json({
        "Embedding Model": config.embedding_model,
        "Vector Dimensions": config.embedding_dimension,
        "Qdrant Status": "Active (Local/Embedded)",
        "BM25 Index Status": "Built",
        "Storage Engine": "SQLite + In-Memory Vector Cache",
        "Total Resumes Indexed": resume_count
    })
