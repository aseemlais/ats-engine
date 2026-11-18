"""
Local Streamlit UI for the spaCy Section-Based ATS Scoring Experiment.
Upload resumes + a Job Description, get section-level scoring, and compare
against the existing baseline — entirely local, zero production changes.
"""
import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ─── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Section-Based ATS Experiment",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Late imports (after sys.path is set) ─────────────────────────────────────
from app.services.jd_parser import JDParserService
from experiments.experiment_config import SectionDetectionConfig, SectionScoringConfig
from experiments.section_scoring_experiment import SectionScoringExperiment


# ─── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Experiment Settings")
    st.caption("All processing is 100% local. Nothing is deployed or saved permanently.")

    st.subheader("Bonus Parameters")
    enable_bonus = st.toggle("Enable Additional-Section Bonus", value=True)
    beta = st.slider(
        "Bonus scaling factor β",
        min_value=0.0, max_value=10.0, value=3.0, step=0.5,
        help="Maximum bonus points added to the core score. β=0 disables the bonus.",
        disabled=not enable_bonus,
    )
    relevance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.0, max_value=0.5, value=0.20, step=0.05,
        help="Minimum combined relevance required to trigger the additional bonus.",
    )

    st.subheader("Section Scoring Mode (Batch 1)")
    st.info("✅ **Independent Scoring Active**\nArbitrary cross-section weights (0.40, 0.30, 0.20, 0.10) are REMOVED. Each section is scored independently on its own evidence.", icon="🎯")

    st.divider()
    st.info("**Safety Note**\nThis app makes zero changes to the deployed backend, Supabase, Qdrant, GitHub, or any remote service.", icon="🔒")


# ─── Main Content ──────────────────────────────────────────────────────────────
st.title("🔬 Local spaCy Section-Based ATS Scoring Experiment")
st.caption("Evaluate whether section-level BM25 + Semantic matching produces more meaningful rankings than whole-resume scoring.")

tab_upload, tab_results, tab_report = st.tabs(["📤 Upload & Run", "📊 Results", "📄 Full Report"])

# ─── Tab 1: Upload ────────────────────────────────────────────────────────────
with tab_upload:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Job Description")
        jd_mode = st.radio("Input method", ["Upload JD file", "Paste JD text", "Use sample JD"], horizontal=True)

        jd_text = ""
        if jd_mode == "Upload JD file":
            jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])
            if jd_file:
                jd_text = jd_file.read().decode("utf-8", errors="replace")
                st.success(f"Loaded: `{jd_file.name}` ({len(jd_text)} chars)")
        elif jd_mode == "Paste JD text":
            jd_text = st.text_area("Paste Job Description text here", height=200, placeholder="Senior Data Analyst\n\nRequired Skills: Python, SQL, Tableau...")
        else:
            sample_jd_path = PROJECT_ROOT / "data" / "resumes" / "JD_Senior_Data_Analyst.txt"
            if sample_jd_path.exists():
                jd_text = sample_jd_path.read_text(encoding="utf-8", errors="replace")
                st.info(f"Using sample JD: `{sample_jd_path.name}`")
                with st.expander("Preview JD"):
                    st.text(jd_text[:800] + ("..." if len(jd_text) > 800 else ""))
            else:
                st.warning("Sample JD not found at `data/resumes/JD_Senior_Data_Analyst.txt`")

    with col_right:
        st.subheader("2. Candidate Resumes")
        resume_mode = st.radio("Input method", ["Upload resume files", "Use sample resumes"], horizontal=True)

        uploaded_files = []
        if resume_mode == "Upload resume files":
            uploaded_files = st.file_uploader(
                "Upload one or more resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                st.success(f"{len(uploaded_files)} resume(s) uploaded.")
        else:
            sample_dir = PROJECT_ROOT / "data" / "resumes"
            if sample_dir.exists():
                sample_resumes = [p for p in sample_dir.iterdir()
                                  if p.suffix.lower() in {".pdf", ".docx", ".txt"}
                                  and not p.name.startswith(("JD_", "jd_"))]
                st.info(f"Using {len(sample_resumes)} sample resume(s) from `data/resumes/`")
                for r in sample_resumes:
                    st.caption(f"📄 `{r.name}`")

    st.divider()

    run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True,
                        disabled=(not jd_text or (resume_mode == "Upload resume files" and not uploaded_files)))

    if not jd_text:
        st.warning("Please provide a Job Description to continue.")
    elif resume_mode == "Upload resume files" and not uploaded_files:
        st.warning("Please upload at least one resume file to continue.")

    if run_btn:
        # Build temp files for uploaded resumes
        tmp_dir = tempfile.mkdtemp(prefix="ats_exp_")
        resume_paths = []

        if resume_mode == "Upload resume files":
            for uf in uploaded_files:
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                resume_paths.append(tmp_path)
        else:
            resume_paths = [str(p) for p in sample_resumes]

        # Build scoring config
        scoring_cfg = SectionScoringConfig(
            BONUS_BETA=beta if enable_bonus else 0.0,
            ENABLE_ADDITIONAL_BONUS=enable_bonus,
            ADDITIONAL_RELEVANCE_THRESHOLD=relevance_threshold,
        )

        with st.status("🚀 Running experiment pipelines...", expanded=True) as status:
            try:
                st.write("📂 Loading and parsing Job Description...")
                exp = SectionScoringExperiment(
                    beta=beta if enable_bonus else 0.0,
                    enable_bonus=enable_bonus,
                )
                # Override the scoring config with UI settings
                exp.scoring_config = scoring_cfg
                exp.experimental_scorer.scoring_config = scoring_cfg

                jd = exp.load_jd(jd_text)
                st.write(f"✅ Parsed JD: **{jd.title}** — {len(jd.required_skills)} required skills, {len(jd.preferred_skills)} preferred skills")

                st.write(f"🔬 Scoring {len(resume_paths)} candidate resume(s) — section-level BM25 + Semantic...")
                comparisons = exp.run_experiment(resume_paths, jd)

                st.write("📝 Generating comparison report...")
                report_md = exp.generate_markdown_report(comparisons, jd)

                status.update(label="✅ Experiment complete!", state="complete", expanded=False)

                # Store results in session state
                st.session_state["comparisons"] = comparisons
                st.session_state["report_md"] = report_md
                st.session_state["jd"] = jd
                st.session_state["exp"] = exp

                st.success("Done! Switch to the **📊 Results** tab to see detailed rankings.")

            except Exception as e:
                status.update(label="❌ Experiment failed.", state="error")
                st.error(f"Error: {type(e).__name__}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# ─── Tab 2: Results ───────────────────────────────────────────────────────────
with tab_results:
    if "comparisons" not in st.session_state:
        st.info("Run the experiment first in the **📤 Upload & Run** tab.")
    else:
        comparisons = st.session_state["comparisons"]
        jd = st.session_state["jd"]

        st.subheader(f"🏆 Candidate Rankings — {jd.title}")
        st.caption(f"Required: `{', '.join(jd.required_skills)}` | Preferred: `{', '.join(jd.preferred_skills) or 'None'}`")

        # ── Summary table ──────────────────────────────────────────────────────
        import pandas as pd

        table_rows = []
        for c in comparisons:
            shift = c.rank_shift
            shift_str = f"▲ +{shift}" if shift > 0 else (f"▼ {shift}" if shift < 0 else "= 0")
            delta_str = f"+{c.delta_score:.2f}%" if c.delta_score >= 0 else f"{c.delta_score:.2f}%"
            # Batch 2: Education eligibility status in summary table
            edu_elig = c.education_eligibility
            edu_status = "—"
            if edu_elig is not None:
                status_val = edu_elig.status.value if hasattr(edu_elig.status, 'value') else str(edu_elig.status)
                edu_icons = {"ELIGIBLE": "🟢", "INELIGIBLE": "🔴", "UNKNOWN": "🟡", "NOT_REQUIRED": "⚪"}
                edu_status = f"{edu_icons.get(status_val, '')} {status_val}"
            # Batch 3: Experience status in summary table
            exp_eval = c.experience_evaluation or getattr(c.experimental_result, 'experience_evaluation', None)
            exp_status = "—"
            if exp_eval is not None:
                x_val = exp_eval.status.value if hasattr(exp_eval.status, 'value') else str(exp_eval.status)
                exp_icons = {"MEETS_REQUIREMENT": "🟢", "BELOW_REQUIREMENT": "🔴", "UNKNOWN": "🟡", "NOT_REQUIRED": "⚪"}
                exp_status = f"{exp_icons.get(x_val, '')} {x_val.replace('_REQUIREMENT', '')}"

            table_rows.append({
                "Exp Rank": c.experimental_rank,
                "Candidate": c.name,
                "File": c.source_file,
                "Exp Score": f"{c.experimental_score:.2f}%",
                "Base Score": f"{c.baseline_score:.2f}%",
                "Δ Score": delta_str,
                "Base Rank": c.baseline_rank,
                "Rank Shift": shift_str,
                "Core Score": f"{c.core_score:.2f}%",
                "Bonus": f"+{c.additional_bonus:.2f}",
                "Edu Filter": edu_status,
                "Exp Filter": exp_status,
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Per-candidate detail cards ─────────────────────────────────────────
        st.subheader("📋 Detailed Section Breakdown")

        for c in comparisons:
            exp_res = c.experimental_result
            ext = c.extraction_result

            delta_color = "green" if c.delta_score >= 0 else "red"
            delta_str = f"+{c.delta_score:.2f}%" if c.delta_score >= 0 else f"{c.delta_score:.2f}%"

            with st.expander(
                f"#{c.experimental_rank} · {c.name} — {c.experimental_score:.2f}%  "
                f"({'⬆' if c.rank_shift > 0 else '⬇' if c.rank_shift < 0 else '='} Δ {delta_str} from baseline)",
                expanded=(c.experimental_rank == 1),
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Experimental Score", f"{c.experimental_score:.2f}%", delta=delta_str)
                col2.metric("Baseline Score", f"{c.baseline_score:.2f}%")
                col3.metric("Core Score", f"{c.core_score:.2f}%")
                col4.metric("Additional Bonus", f"+{c.additional_bonus:.2f} pts")

                # Batch 2: Education Eligibility Status Display
                edu_elig = c.education_eligibility or getattr(exp_res, 'education_eligibility', None)
                if edu_elig is not None:
                    st.divider()
                    st.markdown("**🎓 Education Eligibility (Hard Filter)**")
                    
                    status_colors = {
                        "ELIGIBLE": "🟢",
                        "INELIGIBLE": "🔴",
                        "UNKNOWN": "🟡",
                        "NOT_REQUIRED": "⚪",
                    }
                    status_str = edu_elig.status.value if hasattr(edu_elig.status, 'value') else str(edu_elig.status)
                    icon = status_colors.get(status_str, "⚪")
                    
                    edu_col1, edu_col2, edu_col3 = st.columns(3)
                    edu_col1.metric("Eligibility Status", f"{icon} {status_str}")
                    edu_col2.metric("Matched Degree", edu_elig.matched_degree or "—")
                    edu_col3.metric("Matched Field", edu_elig.matched_field or "—")
                    
                    if status_str == "ELIGIBLE":
                        st.success(f"**Decision:** {edu_elig.decision_reason}")
                    elif status_str == "INELIGIBLE":
                        st.error(f"**Decision:** {edu_elig.decision_reason}")
                    elif status_str == "UNKNOWN":
                        st.warning(f"**Decision:** {edu_elig.decision_reason}")
                        st.caption("⚠️ Flagged for manual verification. Candidate is NOT automatically rejected.")
                    else:
                        st.info(f"**Decision:** {edu_elig.decision_reason}")
                    
                    if edu_elig.jd_requirement_summary:
                        st.caption(f"📋 JD Requirement: `{edu_elig.jd_requirement_summary}`")
                    if edu_elig.candidate_degrees:
                        st.caption(f"🎓 Candidate Degrees: `{', '.join(edu_elig.candidate_degrees)}`")
                    if edu_elig.requires_manual_verification:
                        st.caption("🔍 **Manual verification recommended.**")

                # Batch 3: Experience & Internship Evaluation Display
                exp_eval = c.experience_evaluation or getattr(exp_res, 'experience_evaluation', None)
                if exp_eval is not None:
                    st.divider()
                    st.markdown("**💼 Experience & Internship Evaluation (Batch 3)**")
                    
                    x_status_str = exp_eval.status.value if hasattr(exp_eval.status, 'value') else str(exp_eval.status)
                    x_icons = {
                        "MEETS_REQUIREMENT": "🟢",
                        "BELOW_REQUIREMENT": "🔴",
                        "UNKNOWN": "🟡",
                        "NOT_REQUIRED": "⚪",
                    }
                    x_icon = x_icons.get(x_status_str, "⚪")

                    xc1, xc2, xc3, xc4 = st.columns(4)
                    xc1.metric("Experience Status", f"{x_icon} {x_status_str}")
                    xc2.metric("Verified Duration", f"{exp_eval.verified_total_years:.2f} yrs")
                    xc3.metric("Required Duration", f"{exp_eval.required_years:.2f} yrs")
                    xc4.metric("Relevance-Adjusted", f"{exp_eval.relevance_adjusted_years:.2f} yrs")

                    xr1, xr2, xr3 = st.columns(3)
                    xr1.metric("Duration Ratio", f"{exp_eval.duration_ratio:.2f}")
                    xr2.metric("Eff. Type Multiplier", f"{exp_eval.effective_multiplier:.2f}")
                    xr3.metric("Duties Relevance", f"{exp_eval.duties_relevance:.4f}")

                    if x_status_str == "MEETS_REQUIREMENT":
                        st.success(f"**Decision:** {exp_eval.decision_reason}")
                    elif x_status_str == "BELOW_REQUIREMENT":
                        st.warning(f"**Decision:** {exp_eval.decision_reason}")
                    elif x_status_str == "NOT_REQUIRED":
                        st.info(f"**Decision:** {exp_eval.decision_reason}")
                    else:
                        st.warning(f"**Decision:** {exp_eval.decision_reason}")

                    if exp_eval.fresher_bonus > 0:
                        st.success(f"✨ **Fresher Experience Bonus:** `+{exp_eval.fresher_bonus:.2f} pts` awarded for relevant experience when not mandatory.")

                    if exp_eval.records:
                        with st.expander("📋 View Extracted Experience Records"):
                            rec_rows = []
                            for r in exp_eval.records:
                                dur = f"{r.duration_years:.2f} yrs" if r.duration_years > 0 else f"{r.duration_months:.1f} mos"
                                rec_rows.append({
                                    "Role": r.role_title,
                                    "Organization": r.organization,
                                    "Type": r.exp_type.value,
                                    "Duration": dur,
                                    "Type Mult": r.type_multiplier,
                                    "Duties Rel": round(r.duties_relevance, 4),
                                    "BM25 Norm": round(r.bm25_normalized, 4),
                                    "Semantic Cal": round(r.semantic_calibrated, 4),
                                    "Adj Years": round(r.relevance_adjusted_years, 2),
                                })
                            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

                    if exp_eval.uncertainty_notes:
                        for un in exp_eval.uncertainty_notes:
                            st.caption(f"⚠️ *Uncertainty warning*: {un}")

                st.divider()

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    st.markdown("**Detected Sections**")
                    section_rows = []
                    for s in ext.sections:
                        section_rows.append({
                            "Heading": s.raw_heading[:40],
                            "Category": s.category,
                            "Core?": "✅" if s.is_core else "—",
                            "Chars": len(s.text),
                            "Sentences": len(s.sentences),
                        })
                    if section_rows:
                        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True, height=200)
                    else:
                        st.caption("No sections detected.")

                with col_b:
                    st.markdown("**Independent Section Scores (BM25 + Semantic)**")
                    bm25_rows = []
                    for sec in exp_res.detailed_section_scores:
                        terms = ", ".join(sec.matched_terms[:3]) if sec.matched_terms else ("(None)" if sec.is_core and sec.char_length > 0 else "—")
                        bm25_rows.append({
                            "Section": sec.raw_heading[:20],
                            "Category": sec.category,
                            "BM25 (Norm)": round(sec.bm25_normalized, 4),
                            "BM25 (Raw)": round(sec.bm25_raw, 4),
                            "Semantic (Cal)": round(sec.semantic_calibrated, 4),
                            "Semantic (Cos)": round(sec.semantic_raw_cosine, 4),
                            "Matched Terms": terms,
                        })
                    st.dataframe(pd.DataFrame(bm25_rows), use_container_width=True, hide_index=True)

                    agg_col1, agg_col2 = st.columns(2)
                    agg_col1.metric("Provisional BM25 Bridge", f"{exp_res.bm25_core_aggregate:.4f}")
                    agg_col2.metric("Provisional Semantic Bridge", f"{exp_res.semantic_core_aggregate:.4f}")

                # Batch 4: Skills and Projects Evaluation Display
                sk_eval = exp_res.skills_evaluation
                proj_eval = exp_res.projects_evaluation
                cert_eval = exp_res.certifications_evaluation

                if sk_eval is not None and sk_eval.is_applicable:
                    st.divider()
                    st.markdown("**🛠️ Skills Evaluation (Batch 4)**")
                    sk_c1, sk_c2, sk_c3, sk_c4 = st.columns(4)
                    sk_c1.metric("Overall Skill Score", f"{sk_eval.overall_skill_score:.4f}")
                    sk_c2.metric("Req Coverage", f"{sk_eval.required_coverage:.1%}")
                    sk_c3.metric("Pref Coverage", f"{sk_eval.preferred_coverage:.1%}")
                    sk_c4.metric("Matched Req", f"{len(sk_eval.matched_required)}/{len(sk_eval.required_skills)}")

                    if sk_eval.matched_required:
                        st.caption(f"✅ **Matched Required**: `{', '.join(sk_eval.matched_required)}`")
                    if sk_eval.missing_required:
                        st.caption(f"❌ **Missing Required**: `{', '.join(sk_eval.missing_required)}`")
                    if sk_eval.matched_preferred:
                        st.caption(f"🌟 **Matched Preferred**: `{', '.join(sk_eval.matched_preferred)}`")
                    if sk_eval.deduplication_warnings:
                        for dw in sk_eval.deduplication_warnings:
                            st.caption(f"🔁 *Deduplication*: {dw}")

                if proj_eval is not None and proj_eval.is_applicable:
                    st.divider()
                    st.markdown("**📁 Projects Evaluation (Batch 4)**")
                    pr_c1, pr_c2, pr_c3 = st.columns(3)
                    pr_c1.metric("Project Score", f"{proj_eval.overall_project_score:.4f}")
                    pr_c2.metric("Unique Projects", proj_eval.unique_projects_count)
                    pr_c3.metric("Duplicate Projects", proj_eval.duplicate_projects_count)

                    if proj_eval.projects:
                        with st.expander("📋 View Evaluated Projects"):
                            p_rows = []
                            for p in proj_eval.projects:
                                p_rows.append({
                                    "Project": p.title,
                                    "Relevance Score": p.relevance_score,
                                    "BM25 Norm": p.bm25_normalized,
                                    "Semantic Cal": p.semantic_calibrated,
                                    "Duplicate?": "⚠️ Yes" if p.is_duplicate else "No",
                                })
                            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

                if cert_eval is not None and cert_eval.is_applicable:
                    st.divider()
                    st.markdown("**📜 Certifications Evaluation (Batch 4)**")
                    st.caption(f"Requirement Level: `{cert_eval.requirement_level}` | Direct Evidence: `{'Yes' if cert_eval.has_direct_evidence else 'No'}` | Score: `{cert_eval.score:.4f}`")

                # Core dimension scores
                st.divider()
                st.markdown("**Core ATS Dimension Scores (0–1)**")
                dim_cols = st.columns(5)
                dims = ["semantic", "bm25", "skill", "experience", "certification"]
                for i, dim in enumerate(dims):
                    val = exp_res.core_dimension_scores.get(dim, 0.0)
                    dim_cols[i].metric(dim.capitalize(), f"{val:.4f}")

                # Additional bonus detail
                bb = exp_res.additional_bonus_breakdown
                if exp_res.additional_bonus > 0:
                    st.divider()
                    st.markdown("**Additional-Section Bonus Breakdown**")
                    st.success(
                        f"Bonus awarded: **+{exp_res.additional_bonus:.2f} pts** "
                        f"(Relevance: {exp_res.additional_relevance:.4f} ≥ threshold {st.session_state['exp'].scoring_config.ADDITIONAL_RELEVANCE_THRESHOLD})"
                    )
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Semantic Relevance", f"{bb.get('semantic_relevance', 0):.4f}")
                    bc2.metric("Lexical Relevance", f"{bb.get('lexical_relevance', 0):.4f}")
                    bc3.metric("Combined Relevance", f"{bb.get('combined_relevance', 0):.4f}")

                    new_skills = bb.get("relevant_new_skills", [])
                    dedup_count = bb.get("deduplicated_core_skills_count", 0)
                    if new_skills:
                        st.caption(f"✨ New relevant skills from non-core sections: `{', '.join(new_skills)}`")
                    if dedup_count > 0:
                        st.caption(f"🔁 {dedup_count} skills already credited in core sections were excluded (no double-counting).")

                # Extraction notes
                if exp_res.notes:
                    with st.expander("📌 Extraction & Processing Notes"):
                        for note in exp_res.notes:
                            st.caption(f"• {note}")


# ─── Tab 3: Full Markdown Report ──────────────────────────────────────────────
with tab_report:
    if "report_md" not in st.session_state:
        st.info("Run the experiment first in the **📤 Upload & Run** tab.")
    else:
        report_md = st.session_state["report_md"]
        jd = st.session_state["jd"]

        st.subheader("📄 Full Comparison Report")
        dl_col, _ = st.columns([1, 4])
        dl_col.download_button(
            label="⬇ Download Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name="section_scoring_comparison_report.md",
            mime="text/markdown",
        )
        st.divider()
        st.markdown(report_md)
