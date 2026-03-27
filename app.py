# app.py
import re
import json
import streamlit as st

from core.pipeline_loader import init_pipeline
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import render_user_message, render_bot_message, show_thinking
from ui.json_renderer import JSONRenderer
from rag.pipeline_diagnostics import DiagnosticPipeline

setup_page()
inject_custom_css()
page_title()


def normalize_output(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return t.strip()


# ── Init pipeline once ────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

    # Generate unanswerable questions ONCE at startup for M5 (safe refusal).
    # Cached on the credibility evaluator so every query reuses them — no
    # repeated LLM calls per answer.
    try:
        evaluator = st.session_state.pipeline.credibility_evaluator
        with st.spinner("Initialising evaluation cache (one-time setup)..."):
            evaluator.unanswerable_cache = evaluator.generate_unanswerable_questions(n=10)
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"M5 cache: {len(evaluator.unanswerable_cache)} unanswerable questions generated"
        )
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning(f"M5 cache generation failed: {_e}")
        st.session_state.pipeline.credibility_evaluator.unanswerable_cache = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("⚙️ **Controls**")
    mode = st.radio(
        "Answer style",
        ["Short", "Standard", "Evidence"],
        index=1,
        help="Short = quick. Standard = fuller. Evidence = includes verbatim quotes.",
    )

    st.markdown("---")

    # ── Query Retrieval Analysis ──────────────────────────────────────────────
    st.markdown("🔍 **Diagnostics**")

    with st.expander("Query Retrieval Analysis", expanded=False):
        st.caption("See what documents are retrieved for any query")

        diagnostic_query = st.text_input(
            "Test Query",
            placeholder="Enter a question to diagnose...",
            key="diagnostic_query_input",
            help="Analyze which chunks are retrieved for this query",
        )

        chunks_k = st.slider(
            "Chunks per sub-query",
            min_value=5,
            max_value=20,
            value=10,
            key="diagnostic_k",
            help="Number of chunks to retrieve for each generated sub-query",
        )

        run_diagnostic = st.button(
            "🔍 Run Diagnostic",
            use_container_width=True,
            type="secondary",
            key="run_diagnostic_btn",
        )

        if run_diagnostic and diagnostic_query:
            with st.spinner("Running diagnostic analysis..."):
                try:
                    diagnosis = st.session_state.pipeline.diagnose_retrieval(
                        diagnostic_query, k=chunks_k
                    )
                    st.session_state.diagnostic_results = diagnosis
                    st.success("✅ Diagnostic complete! See results below.")
                except Exception as e:
                    st.error(f"❌ Diagnostic failed: {str(e)}")
                    st.exception(e)
        elif run_diagnostic and not diagnostic_query:
            st.warning("⚠️ Please enter a query to diagnose")

    st.markdown("---")

    # ── Full Pipeline Diagnostic ──────────────────────────────────────────────
    st.markdown("🧪 **Full Pipeline Diagnostic**")

    with st.expander("Full Pipeline Trace", expanded=False):
        st.caption(
            "Traces every step: classification → retrieval → reranking → "
            "context → LLM answer → faithfulness. Downloads full JSON report."
        )
        full_diag_query = st.text_input(
            "Question to diagnose",
            placeholder="Paste the question that gave a bad answer...",
            key="full_diag_query_input",
        )
        if st.button(
            "🧪 Run Full Diagnostic",
            use_container_width=True,
            type="secondary",
            key="run_full_diag_btn",
        ):
            if not full_diag_query.strip():
                st.warning("⚠️ Please enter a question first.")
            else:
                with st.spinner(
                    "Running full pipeline trace — classification, retrieval, "
                    "reranking, LLM, faithfulness... (~1–2 min)"
                ):
                    try:
                        diag   = DiagnosticPipeline(st.session_state.pipeline)
                        report = diag.run(question=full_diag_query.strip())
                        st.session_state.full_diag_report = report
                        st.success("✅ Diagnostic complete. See results below.")
                    except Exception as e:
                        st.error(f"❌ Diagnostic failed: {e}")
                        st.exception(e)

    st.markdown("---")

    # ── Credibility Check ─────────────────────────────────────────────────────
    st.markdown(
        '<span title="Check claims from other sources by verifying them against '
        'the indexed thyroid cancer papers.">'
        "✅ **Credibility Check**"
        "</span>",
        unsafe_allow_html=True,
    )
    credibility_on = st.toggle(
        "Enable",
        value=False,
        help="When enabled, paste a claim and press the button to run the check.",
    )

    claim_text = ""
    run_cred   = False
    if credibility_on:
        claim_text = st.text_area(
            "Paste the claim to verify",
            placeholder="Example: Radioiodine ablation decreases local recurrence risk in papillary thyroid cancer.",
            height=140,
        )
        run_cred = st.button("Run credibility check", use_container_width=True)

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        for key in ("diagnostic_results", "full_diag_report"):
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
if "disclaimer_shown" not in st.session_state:
    st.session_state.disclaimer_shown = True
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <b>Disclaimer:</b> This tool is for research/education only and does "
        "not provide medical advice. For clinical decisions, consult guidelines "
        "and qualified healthcare professionals."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RETRIEVAL DIAGNOSTIC RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if "diagnostic_results" in st.session_state:
    diagnosis = st.session_state.diagnostic_results

    with st.container():
        st.markdown("### 🔍 Diagnostic Results")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Question:** {diagnosis['original_question']}")
        with col2:
            if st.button("❌ Close", key="close_diagnostic"):
                del st.session_state.diagnostic_results
                st.rerun()

        st.markdown("**Generated Sub-Queries:**")
        for i, sq in enumerate(diagnosis["sub_queries_generated"], 1):
            st.markdown(f"{i}. `{sq}`")

        st.markdown("---")

        for idx, result in enumerate(diagnosis["retrieval_results"], 1):
            with st.expander(
                f"📊 Sub-Query {idx}: {result['query'][:80]}... "
                f"({result['chunks_found']} chunks found)",
                expanded=(idx == 1),
            ):
                if result["sample_chunks"]:
                    for chunk in result["sample_chunks"]:
                        st.markdown(
                            f"**Rank {chunk['rank']}** | "
                            f"<span class='diagnostic-score'>Score: {chunk['score']:.4f}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"📄 {chunk['title']} ({chunk['year']}) | "
                            f"PMID: {chunk['pmid']} | "
                            f"Evidence Level: {chunk['evidence_level']}"
                        )
                        st.text_area(
                            f"Text Preview (Rank {chunk['rank']})",
                            value=chunk["text_preview"],
                            height=120,
                            disabled=True,
                            key=f"diag_chunk_{idx}_{chunk['rank']}",
                        )
                        if chunk != result["sample_chunks"][-1]:
                            st.markdown("---")
                else:
                    st.warning("⚠️ No chunks retrieved for this sub-query")

        st.download_button(
            label="📥 Download Full Diagnostic (JSON)",
            data=json.dumps(diagnosis, indent=2),
            file_name="diagnostic_analysis.json",
            mime="application/json",
            help="Download complete diagnostic data for further analysis",
            use_container_width=True,
        )
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE DIAGNOSTIC RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if "full_diag_report" in st.session_state:
    report          = st.session_state.full_diag_report
    summary         = report.get("step_12_final_output", {})
    issues          = summary.get("issues_detected", [])
    recs            = summary.get("recommendations", [])
    quality         = summary.get("answer_quality", {})
    retrieval_stats = summary.get("retrieval_stats", {})

    with st.container():
        st.markdown("### 🧪 Full Pipeline Diagnostic Report")

        col_q, col_close = st.columns([4, 1])
        with col_q:
            st.markdown(f"**Question:** {report.get('meta', {}).get('question', 'N/A')}")
        with col_close:
            if st.button("❌ Close", key="close_full_diag"):
                del st.session_state.full_diag_report
                st.rerun()

        with st.expander("📌 Step 2 — Classification", expanded=True):
            s2 = report.get("step_2_classification", {})
            col1, col2 = st.columns(2)
            col1.metric("Question Type", s2.get("final_type", "N/A"))
            col2.metric("Layer Fired",   s2.get("layer_fired", "N/A"))
            st.markdown(f"**Layer 1 keyword match:** `{s2.get('layer_1_keyword_result') or 'no match → pass to LLM'}`")
            st.markdown(f"**Layer 2 LLM raw output:** `{s2.get('layer_2_llm_raw_output', 'N/A')}`")
            st.markdown(f"**Layer 2 validated category:** `{s2.get('layer_2_llm_validated_category', 'N/A')}`")
            if s2.get("layer_2_llm_fallback_triggered"):
                st.warning(f"⚠️ LLM returned invalid category — fell back to 'definition'. Detail: {s2.get('layer_2_detail', '')}")
            for detail in s2.get("layer_3_detail", []):
                st.info(f"🔄 Layer 3 safety net fired: {detail}")

        with st.expander("🔀 Step 3 — Query Expansion", expanded=True):
            s3 = report.get("step_3_query_expansion", {})
            st.markdown(f"**Expansion source:** {s3.get('expansion_source', 'N/A')}")
            st.markdown(f"**Total sub-queries:** {s3.get('total_sub_queries', 0)}")
            if s3.get("llm_parse_error"):
                st.error(f"LLM JSON parse error: {s3['llm_parse_error']}")
            if s3.get("llm_error"):
                st.error(f"LLM error: {s3['llm_error']}")
            llm_qs      = set(s3.get("queries_from_llm", []))
            fallback_qs = set(s3.get("queries_added_by_domain_fallback", []))
            original_q  = report.get("meta", {}).get("question", "")
            st.markdown("**Sub-queries sent to vector store:**")
            for i, q in enumerate(s3.get("sub_queries", []), 1):
                if q == original_q:
                    lbl = "📝 original question"
                elif q in llm_qs:
                    lbl = "🤖 LLM generated"
                elif q in fallback_qs:
                    lbl = "📎 domain fallback added"
                else:
                    lbl = ""
                st.markdown(f"  {i}. `{q}` — *{lbl}*")
            if not fallback_qs:
                st.warning("⚠️ No domain-specific fallback queries were added.")

        with st.expander("📦 Step 4 — Retrieval (Bi-encoder)", expanded=False):
            s4 = report.get("step_4_retrieval", {})
            col1, col2 = st.columns(2)
            col1.metric("Total chunks retrieved", s4.get("total_chunks_retrieved", 0))
            col2.metric("Chunks per sub-query",   s4.get("chunks_per_query", "N/A"))
            for q_result in s4.get("per_query_results", []):
                st.markdown(f"**Sub-query {q_result['query_index']}:** `{q_result['query']}`")
                st.caption(f"Chunks returned: {q_result['chunks_returned']}")
                for chunk in q_result.get("top_5_preview", []):
                    st.markdown(
                        f"&nbsp;&nbsp;Rank **{chunk['rank']}** | "
                        f"Score: `{chunk['score']:.4f}` | "
                        f"Evidence L{chunk['evidence_level']} | "
                        f"PMID: {chunk['pmid']}",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"  📄 {chunk['title']}")
                    st.text_area(
                        f"Text — sub-query {q_result['query_index']}, rank {chunk['rank']}",
                        value=chunk.get("text_preview", ""),
                        height=100,
                        disabled=True,
                        key=f"fd_chunk_{q_result['query_index']}_{chunk['rank']}",
                    )
                st.markdown("---")

        with st.expander("🔄 Steps 5–6 — Deduplication & Cross-encoder Reranking", expanded=False):
            s5 = report.get("step_5_deduplication", {})
            s6 = report.get("step_6_reranking", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Before dedup",       s5.get("before", 0))
            col2.metric("After dedup",        s5.get("after", 0))
            col3.metric("Duplicates removed", s5.get("removed_count", 0))
            col4.metric("After reranking",    s6.get("output_chunk_count", 0))
            st.markdown(f"**Bi-encoder score range:** `{s6.get('bi_encoder_score_min','N/A')}` → `{s6.get('bi_encoder_score_max','N/A')}`")
            st.markdown(f"**Cross-encoder score range:** `{s6.get('cross_encoder_score_min','N/A')}` → `{s6.get('cross_encoder_score_max','N/A')}`")
            st.markdown("**Top 5 chunks after reranking:**")
            for chunk in s6.get("all_reranked_chunks", [])[:5]:
                delta = chunk.get("rank_delta")
                arrow = f"⬆️ +{delta}" if delta and delta > 0 else (f"⬇️ {delta}" if delta and delta < 0 else "➡️ 0")
                st.markdown(
                    f"  Rank **{chunk['new_rank']}** *(was {chunk['original_rank']})* {arrow} | "
                    f"CE: `{chunk['cross_encoder_score']:.4f}` | "
                    f"BE: `{chunk['bi_encoder_score']:.4f}` | "
                    f"L{chunk['evidence_level']} | {chunk['title'][:70]}"
                )
            sig = s6.get("significant_rank_changes", [])
            if sig:
                st.markdown("**Notable rank changes (≥5 positions):**")
                for c in sig[:5]:
                    direction = "⬆️ promoted" if (c.get("rank_delta") or 0) > 0 else "⬇️ demoted"
                    st.markdown(
                        f"  {direction} | rank {c['original_rank']} → {c['new_rank']} "
                        f"(Δ{abs(c.get('rank_delta') or 0)}) | {c['title'][:60]}"
                    )

        with st.expander("📋 Step 7 — Context Construction", expanded=False):
            s7 = report.get("step_7_context", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Sources in context", s7.get("source_count", 0))
            col2.metric("Context chars",      s7.get("total_context_chars", 0))
            col3.metric("Truncated", "Yes ⚠️" if s7.get("truncated_at_limit") else "No ✅")
            if s7.get("truncated_at_limit"):
                st.warning("⚠️ Context hit the 10,000 char limit. Some retrieved content was not sent to the LLM.")
            st.markdown("**Source map (what the LLM sees):**")
            for tag, meta in s7.get("source_map", {}).items():
                st.markdown(
                    f"  `{tag}` | L{meta.get('evidence_level')} "
                    f"({meta.get('evidence_level_description','N/A')}) | "
                    f"CE score: `{meta.get('cross_encoder_score',0):.4f}` | "
                    f"PMID: {meta.get('pmid','N/A')} | {meta.get('title','N/A')[:60]}"
                )
            with st.expander("View full tagged context sent to LLM", expanded=False):
                st.text_area("Full context", value=s7.get("full_tagged_context",""), height=400, disabled=True, key="fd_context_text")

        with st.expander("💬 Steps 8–9 — LLM Prompt & Raw Response", expanded=False):
            s8 = report.get("step_8_llm_prompt", {})
            s9 = report.get("step_9_raw_response", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Template used", s8.get("question_type", "N/A"))
            col2.metric("Prompt chars",  s8.get("prompt_char_count", 0))
            col3.metric("Prompt words",  s8.get("prompt_word_count", 0))
            with st.expander("View full LLM prompt", expanded=False):
                st.text_area("Full prompt", value=s8.get("full_prompt",""), height=400, disabled=True, key="fd_prompt_text")
            st.markdown("---")
            if s9.get("json_parse_success"):
                st.success("✅ LLM response parsed as valid JSON")
            else:
                st.error(f"❌ JSON parse failed: {s9.get('json_parse_error','N/A')}")
            col1, col2 = st.columns(2)
            col1.metric("Response chars",    s9.get("response_char_count", 0))
            col2.metric("Had markdown fence", "Yes" if s9.get("had_markdown_fence") else "No")
            with st.expander("View raw LLM response", expanded=False):
                st.text_area("Raw response", value=s9.get("full_raw_response",""), height=400, disabled=True, key="fd_raw_response_text")

        with st.expander("✅ Step 10 — Parsed Answer Analysis", expanded=True):
            s10       = report.get("step_10_parsed_answer", {})
            populated  = s10.get("populated_sections", [])
            null_secs  = s10.get("null_sections", [])
            violations = s10.get("placeholder_violations", [])
            col1, col2, col3 = st.columns(3)
            col1.metric("Sections populated",     len(populated))
            col2.metric("Sections null/empty",    len(null_secs))
            col3.metric("Placeholder violations", len(violations))
            if populated:
                st.success(f"✅ Populated: {', '.join(populated)}")
            if null_secs:
                st.error(f"❌ Empty sections: {', '.join(null_secs)}")
            for v in violations:
                st.warning(f"⚠️ Schema violation: {v}")
            overview = s10.get("overview_text", "")
            if overview and overview != "null":
                st.markdown(f"**Overview:** {overview[:300]}...")
            if s10.get("section_details"):
                st.markdown("**Per-section breakdown:**")
                for sec in s10["section_details"]:
                    icon = "✅" if sec["is_populated"] else "❌"
                    st.markdown(f"  {icon} **{sec['header']}** | fields: {', '.join(sec['fields_present']) or 'none'}")

        with st.expander("🔬 Step 11 — Faithfulness Evaluation", expanded=True):
            s11 = report.get("step_11_faithfulness", {})
            if s11.get("skipped"):
                st.warning(f"Skipped — {s11.get('reason','N/A')}")
            else:
                score = s11.get("overall_score", 0)
                label = s11.get("label", "N/A")
                if label == "High":
                    st.success(f"✅ Faithfulness: **{label}** (score: {score:.2f})")
                elif label == "Medium":
                    st.warning(f"⚠️ Faithfulness: **{label}** (score: {score:.2f})")
                else:
                    st.error(f"❌ Faithfulness: **{label}** (score: {score:.2f})")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total statements", s11.get("total_statements_extracted", 0))
                col2.metric("Evaluated",        s11.get("evaluated_statements", 0))
                col3.metric("✅ Faithful",       s11.get("faithful_count", 0))
                col4.metric("⚠️ Partial",        s11.get("partial_count", 0))
                col5.metric("❌ Not faithful",   s11.get("not_faithful_count", 0))
                st.markdown("**Per-statement results:**")
                for detail in s11.get("statement_details", []):
                    verdict = detail.get("verdict", "")
                    score_d = detail.get("score")
                    icon    = "✅" if verdict == "FAITHFUL" else ("⚠️" if verdict == "PARTIAL" else ("⏭️" if "SKIPPED" in verdict else "❌"))
                    score_str = f"({score_d:.2f})" if score_d is not None else ""
                    st.markdown(
                        f"{icon} **{verdict}** {score_str}  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{detail['statement'][:160]}...",
                        unsafe_allow_html=True,
                    )

        with st.expander("📊 Step 12 — Summary & Issues", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Question type",    summary.get("classification", {}).get("final_type", "N/A"))
            col2.metric("Stage 1 chunks",   retrieval_stats.get("chunks_stage_1_bi_encoder", 0))
            col3.metric("After reranking",  retrieval_stats.get("chunks_after_reranking", 0))
            col4.metric("Sources in answer",retrieval_stats.get("sources_in_final_context", 0))
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence",        quality.get("evidence_confidence", "N/A"))
            col2.metric("Faithfulness",      f"{quality.get('faithfulness_label','N/A')} ({quality.get('faithfulness_score','N/A')})")
            col3.metric("Classification layer", summary.get("classification", {}).get("layer_fired", "N/A"))
            st.markdown("---")
            if issues:
                st.markdown("**⚠️ Issues detected:**")
                for issue in issues:
                    st.error(f"• {issue}")
            else:
                st.success("✅ No issues detected")
            if recs:
                st.markdown("**💡 Recommendations:**")
                for rec in recs:
                    st.info(f"→ {rec}")

        st.download_button(
            label="📥 Download Full Diagnostic Report (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name="full_pipeline_diagnostic.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"], unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_mode_hint(user_text: str) -> str:
    if mode == "Short":
        return f"{user_text}\n\n(Answer in short mode.)"
    if mode == "Evidence":
        return f"{user_text}\n\n(Include verbatim evidence quotes.)"
    return user_text


def format_answer_from_json(result: dict) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"

    json_response         = result.get("json_response")
    sources               = result.get("sources", {})
    confidence            = result.get("confidence", {})
    faithfulness          = result.get("faithfulness", {})
    question_type         = result.get("question_type", "definition")
    credibility_scorecard = result.get("credibility_scorecard", {})

    if not json_response:
        return "Unable to generate a structured response."

    renderer = JSONRenderer(
        json_response,
        sources,
        confidence,
        faithfulness,
        credibility_scorecard=credibility_scorecard,
    )

    answer_md     = renderer.render(question_type)
    confidence_md = renderer.render_confidence()   # returns "" — kept for compat
    sources_md    = renderer.render_sources()       # scorecard renders here

    return f"{answer_md}\n\n{confidence_md}\n\n{sources_md}\n"


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT HANDLING
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about thyroid cancer…")

# Sidebar credibility check
if run_cred and claim_text.strip():
    combined = apply_mode_hint(f"CREDIBILITY_CHECK: {claim_text.strip()}")

    st.session_state.messages.append(
        {"role": "user", "content": f"[Credibility Check] {claim_text.strip()}"}
    )
    render_user_message(f"[Credibility Check] {claim_text.strip()}")

    thinking_ph = show_thinking()
    result      = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = normalize_output(format_answer_from_json(result))
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)

elif user_input:
    ui_text  = user_input.strip()
    combined = ui_text

    if ui_text.lower().startswith("[credibility check]"):
        combined = "CREDIBILITY_CHECK: " + ui_text[len("[credibility check]"):].strip()

    combined = apply_mode_hint(combined)

    st.session_state.messages.append({"role": "user", "content": ui_text})
    render_user_message(ui_text)

    thinking_ph = show_thinking()
    result      = st.session_state.pipeline.answer(combined, chat_history=st.session_state.messages)
    thinking_ph.empty()

    answer = normalize_output(format_answer_from_json(result))
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
