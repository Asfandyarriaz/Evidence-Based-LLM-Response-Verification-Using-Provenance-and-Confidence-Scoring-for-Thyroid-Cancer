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


# ✅ CHANGE 3 — UPDATED FUNCTION
def format_answer_from_json(result: dict) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"

    json_response         = result.get("json_response")
    sources               = result.get("sources", {})
    confidence            = result.get("confidence", {})
    faithfulness          = result.get("faithfulness", {})
    question_type         = result.get("question_type", "definition")
    credibility_scorecard = result.get("credibility_scorecard", {})  # ← NEW

    if not json_response:
        return "Unable to generate a structured response."

    renderer = JSONRenderer(
        json_response,
        sources,
        confidence,
        faithfulness,
        credibility_scorecard=credibility_scorecard,  # ← NEW
    )

    answer_md     = renderer.render(question_type)
    confidence_md = renderer.render_confidence()
    sources_md    = renderer.render_sources()

    return f"{answer_md}\n\n{confidence_md}\n\n{sources_md}\n"


# ─────────────────────────────────────────────────────────────────────────────
# INIT SESSION
# ─────────────────────────────────────────────────────────────────────────────

# ✅ CHANGE 2 — UPDATED PIPELINE INIT
if "pipeline" not in st.session_state:
    st.session_state.pipeline = init_pipeline()

    # Cache unanswerable questions ONCE (for M5 safe refusal)
    try:
        evaluator = st.session_state.pipeline.credibility_evaluator

        with st.spinner("Initialising evaluation cache (one-time setup)..."):
            evaluator.unanswerable_cache = evaluator.generate_unanswerable_questions(n=10)

        import logging
        logging.getLogger(__name__).info(
            f"Cached {len(evaluator.unanswerable_cache)} unanswerable questions"
        )

    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning(f"M5 cache generation failed: {_e}")
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
    )

    st.markdown("---")

    # (Your existing diagnostics UI remains unchanged here)


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
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about thyroid cancer…")

if user_input:
    ui_text = user_input.strip()

    # Store user message
    st.session_state.messages.append({"role": "user", "content": ui_text})
    render_user_message(ui_text)

    # Show thinking spinner
    thinking_ph = show_thinking()

    # Run pipeline
    result = st.session_state.pipeline.answer(
        apply_mode_hint(ui_text),
        chat_history=st.session_state.messages
    )

    # ❗ IMPORTANT: STORE RESULT (needed for evaluation system)
    st.session_state.last_pipeline_result   = result
    st.session_state.last_pipeline_question = ui_text

    thinking_ph.empty()

    # Format + render answer
    answer = normalize_output(format_answer_from_json(result))

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
