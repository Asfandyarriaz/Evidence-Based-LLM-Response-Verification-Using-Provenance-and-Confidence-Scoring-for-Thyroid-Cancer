import re
import json
import streamlit as st

# Existing imports
from rag.credibility_evaluator import CredibilityEvaluator
from core.pipeline_loader import init_pipeline
from ui.layout import setup_page, inject_custom_css, page_title
from ui.components import render_user_message, render_bot_message, show_thinking
from ui.json_renderer import JSONRenderer
from rag.pipeline_diagnostics import DiagnosticPipeline

setup_page()
inject_custom_css()
page_title()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize_output(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return t.strip()


def apply_mode_hint(user_text: str) -> str:
    if mode == "Short":
        return f"{user_text}\n\n(Answer in short mode.)"
    if mode == "Evidence":
        return f"{user_text}\n\n(Include verbatim evidence quotes.)"
    return user_text


def format_answer_from_json(result: dict) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"

    json_response = result.get("json_response")
    sources       = result.get("sources", {})
    confidence    = result.get("confidence", {})
    faithfulness  = result.get("faithfulness", {})
    question_type = result.get("question_type", "definition")

    if not json_response:
        return "Unable to generate a structured response."

    renderer = JSONRenderer(json_response, sources, confidence, faithfulness)

    answer_md     = renderer.render(question_type)
    confidence_md = renderer.render_confidence()
    sources_md    = renderer.render_sources()

    return f"{answer_md}\n\n{confidence_md}\n\n{sources_md}\n"


# ─────────────────────────────────────────────────────────────────────────────
# INIT SESSION
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

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
    )

    st.markdown("---")

    # EXISTING DIAGNOSTICS
    st.markdown("🔍 **Diagnostics**")

    # (keep your existing diagnostic UI here unchanged...)

    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────
    # 🏅 NEW: CREDIBILITY SCORECARD
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("🏅 **Credibility Scorecard**")

    with st.expander("Run Credibility Evaluation", expanded=False):

        eval_mode = st.radio(
            "Evaluation mode",
            ["Single question", "Batch (full test set)"],
        )

        run_consistency_toggle = st.toggle(
            "Include consistency (M6)", value=False
        )

        n_unanswerable = st.slider(
            "Unanswerable questions for M5",
            5, 20, 10
        )

        run_eval_btn = st.button("▶ Run Scorecard", use_container_width=True)

        if run_eval_btn:

            # ── SINGLE ───────────────────────────────────────────────
            if eval_mode == "Single question":
                last_result = st.session_state.get("last_pipeline_result")
                last_question = st.session_state.get("last_pipeline_question")

                if not last_result:
                    st.warning("Ask a question first.")
                else:
                    with st.spinner("Computing scorecard..."):
                        evaluator = CredibilityEvaluator(
                            pipeline=st.session_state.pipeline,
                            llm=st.session_state.pipeline.llm,
                        )

                        unanswerable = evaluator.generate_unanswerable_questions(
                            n=n_unanswerable
                        )

                        scorecard = evaluator.compute_scorecard(
                            question=last_question,
                            result=last_result,
                            run_consistency=run_consistency_toggle,
                            unanswerable_questions=unanswerable,
                        )

                        st.session_state.last_scorecard = scorecard
                        st.success("Done.")

            # ── BATCH ───────────────────────────────────────────────
            else:
                from eval.run_evaluation import DEFAULT_QUESTIONS

                with st.spinner("Running batch..."):
                    evaluator = CredibilityEvaluator(
                        pipeline=st.session_state.pipeline,
                        llm=st.session_state.pipeline.llm,
                    )

                    batch = evaluator.run_batch_evaluation(
                        questions=DEFAULT_QUESTIONS,
                        run_consistency=run_consistency_toggle,
                        n_unanswerable=n_unanswerable,
                    )

                    st.session_state.last_batch_eval = batch
                    st.success("Batch complete.")


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if "last_scorecard" in st.session_state:
    sc = st.session_state.last_scorecard

    st.markdown("### 🏅 Credibility Scorecard")

    st.metric("Total Score", f"{sc.get('total_score')}/100")
    st.metric("Label", sc.get("overall_label"))

    st.json(sc)  # full debug view


if "last_batch_eval" in st.session_state:
    batch = st.session_state.last_batch_eval

    st.markdown("### 📊 Batch Evaluation")
    st.metric("Avg Score", batch["summary"]["avg_total_score"])

    st.json(batch["summary"])


# ─────────────────────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask...")

if user_input:
    ui_text = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": ui_text})
    render_user_message(ui_text)

    thinking = show_thinking()

    result = st.session_state.pipeline.answer(
        apply_mode_hint(ui_text),
        chat_history=st.session_state.messages
    )

    # ⭐ CRITICAL (for scorecard)
    st.session_state.last_pipeline_result   = result
    st.session_state.last_pipeline_question = ui_text

    thinking.empty()

    answer = normalize_output(format_answer_from_json(result))

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
