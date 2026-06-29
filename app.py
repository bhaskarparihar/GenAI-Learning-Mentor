# ============================================================
# app.py - Streamlit Frontend for GenAI Learning Mentor
# ============================================================

import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Learning Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ── Cards / containers ── */
.card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(106,17,203,0.3); }

/* ── Hero title ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(106,17,203,0.3), rgba(37,117,252,0.3));
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.6); }

/* ── Chat bubbles ── */
.chat-user {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    margin: 0.4rem 0;
    color: white;
    max-width: 85%;
    margin-left: auto;
    word-wrap: break-word;
}
.chat-bot {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    margin: 0.4rem 0;
    color: rgba(255,255,255,0.9);
    max-width: 85%;
    word-wrap: break-word;
}

/* ── Quiz option buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(106,17,203,0.4), rgba(37,117,252,0.4));
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    color: white;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    border-color: #a78bfa;
    transform: scale(1.02);
}

/* ── Section headers ── */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #a78bfa;
    border-bottom: 2px solid rgba(167,139,250,0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Correct / incorrect badges ── */
.badge-correct { background: #10b981; color: white; padding: 2px 10px; border-radius: 20px; font-size:0.8rem; }
.badge-wrong   { background: #ef4444; color: white; padding: 2px 10px; border-radius: 20px; font-size:0.8rem; }
.badge-info    { background: #3b82f6; color: white; padding: 2px 10px; border-radius: 20px; font-size:0.8rem; }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    color: white !important;
}
label, p, li, h1, h2, h3, h4, h5, h6, .stMarkdown { color: rgba(255,255,255,0.9) !important; }
.stSlider { color: white !important; }
hr { border-color: rgba(255,255,255,0.15); }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────
def init_state():
    defaults = {
        "provider": "gemini",
        "api_key_set": False,
        "chat_history": [],
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": 0,
        "rag_engine": None,
        "agent": None,
        "docs_loaded": 0,
        "quiz_results_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Helper: load engine + agent ───────────────────────────────
def load_engine_and_agent():
    from rag_engine import RAGEngine
    from learning_agent import LearningAgent

    p = st.session_state["provider"]
    if st.session_state["rag_engine"] is None:
        st.session_state["rag_engine"] = RAGEngine(provider=p)
    if st.session_state["agent"] is None:
        st.session_state["agent"] = LearningAgent(provider=p)


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 GenAI Learning Mentor")
    st.markdown("---")

    # ── API setup ──────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")
    provider = st.selectbox(
        "LLM Provider",
        ["gemini", "openai"],
        index=0,
        key="provider_select",
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here…",
        help="Gemini: AIza...  |  OpenAI: sk-...",
    )

    if st.button("🔑 Set API Key", use_container_width=True):
        if api_key:
            if provider == "gemini":
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key
            st.session_state["provider"] = provider
            st.session_state["api_key_set"] = True
            st.session_state["rag_engine"] = None  # reset on provider change
            st.session_state["agent"] = None
            load_engine_and_agent()
            st.success("✅ API key configured!")
        else:
            st.error("Please enter an API key.")

    st.markdown("---")

    # ── Upload course notes ────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    uploaded = st.file_uploader(
        "Upload Course Notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your lecture notes, textbooks, or any study material.",
    )

    if uploaded and st.session_state["api_key_set"]:
        if st.button("📥 Ingest Documents", use_container_width=True):
            load_engine_and_agent()
            total = 0
            with st.spinner("Processing documents…"):
                for f in uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        chunks = st.session_state["rag_engine"].ingest_pdf(tmp_path)
                        total += chunks
                    finally:
                        os.unlink(tmp_path)
            st.session_state["docs_loaded"] += total
            st.success(f"✅ {total} chunks indexed from {len(uploaded)} file(s)!")

    # ── Manual text input ──────────────────────────────────
    manual_text = st.text_area(
        "Or paste notes directly:",
        height=100,
        placeholder="Paste any text content here…",
    )
    if st.button("📝 Ingest Text", use_container_width=True) and manual_text:
        if st.session_state["api_key_set"]:
            load_engine_and_agent()
            chunks = st.session_state["rag_engine"].ingest_text(manual_text)
            st.session_state["docs_loaded"] += chunks
            st.success(f"✅ {chunks} text chunks indexed!")
        else:
            st.warning("Set your API key first.")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        if st.session_state["rag_engine"]:
            st.session_state["rag_engine"].clear_store()
            st.session_state["docs_loaded"] = 0
            st.success("Knowledge base cleared.")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    kb_status = "✅ Ready" if st.session_state.get("docs_loaded", 0) > 0 else "⚠️ Empty"
    st.markdown(f"**Knowledge Base:** {kb_status}")
    st.markdown(f"**Chunks Indexed:** `{st.session_state.get('docs_loaded', 0)}`")
    st.markdown(f"**Messages:** `{len(st.session_state.get('chat_history', []))}`")


# ════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🎓 GenAI Learning Mentor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Your AI-powered adaptive tutor — ask anything, take quizzes, get personalized study plans</div>',
    unsafe_allow_html=True,
)

if not st.session_state["api_key_set"]:
    st.info("👈 **Get started:** Enter your API key in the sidebar and upload your course notes.")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Tutor Chat", "📝 Quiz", "📅 Study Plan", "🔍 Weak Areas", "❓ Practice Questions"]
)


# ════════════════════════════════════════════════════════════
#  TAB 1 – Adaptive Tutor Chat
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">💬 Adaptive Tutor Chat</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your uploaded course material. The mentor uses RAG to find relevant context.")

    # Display history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask your question…",
            placeholder="e.g. Explain gradient descent in simple terms.",
        )
        send = st.form_submit_button("Send 🚀", use_container_width=True)

    if send and user_q:
        if not st.session_state["api_key_set"]:
            st.warning("Please set your API key first.")
        else:
            load_engine_and_agent()
            retriever = st.session_state["rag_engine"].get_retriever()
            with st.spinner("Mentor is thinking…"):
                answer = st.session_state["agent"].ask_tutor(user_q, retriever)
            st.session_state["chat_history"].append({"role": "user", "content": user_q})
            st.session_state["chat_history"].append({"role": "bot", "content": answer})
            st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        if st.session_state.get("agent"):
            st.session_state["agent"].clear_history()
        st.rerun()


# ════════════════════════════════════════════════════════════
#  TAB 2 – Quiz Generator
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📝 Quiz Generator</div>', unsafe_allow_html=True)
    st.markdown("Test your understanding with AI-generated MCQs drawn from your course notes.")

    col_a, col_b = st.columns(2)
    with col_a:
        quiz_topic = st.text_input("Topic (optional)", placeholder="e.g. Neural Networks")
    with col_b:
        num_q = st.slider("Number of Questions", 3, 10, 5)

    if st.button("⚡ Generate Quiz", use_container_width=True):
        if not st.session_state["api_key_set"]:
            st.warning("Set API key first.")
        else:
            load_engine_and_agent()
            retriever = st.session_state["rag_engine"].get_retriever()
            with st.spinner("Generating quiz…"):
                qs = st.session_state["agent"].generate_quiz(
                    retriever,
                    topic=quiz_topic or "general",
                    num_questions=num_q,
                )
            st.session_state["quiz_questions"] = qs
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.rerun()

    if st.session_state["quiz_questions"]:
        st.markdown("---")
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state["quiz_questions"]):
                st.markdown(f"**Q{i+1}. {q.get('question', '')}**")
                opts = q.get("options", {})
                if opts:
                    choice = st.radio(
                        f"Select answer for Q{i+1}",
                        options=list(opts.keys()),
                        format_func=lambda k, o=opts: f"{k}) {o[k]}",
                        key=f"q_{i}",
                        label_visibility="collapsed",
                    )
                    st.session_state["quiz_answers"][i] = choice
                st.markdown("")

            submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True)

        if submitted:
            score = 0
            results_text = ""
            for i, q in enumerate(st.session_state["quiz_questions"]):
                correct = q.get("answer", "")
                selected = st.session_state["quiz_answers"].get(i, "")
                is_correct = selected == correct

                if is_correct:
                    score += 1
                    st.markdown(f'✅ **Q{i+1}:** Correct! <span class="badge-correct">✓</span>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'❌ **Q{i+1}:** You chose **{selected}**, correct is **{correct}** <span class="badge-wrong">✗</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(f"*{q.get('explanation', '')}*")
                st.markdown("---")
                results_text += f"Q{i+1}: {'Correct' if is_correct else f'Wrong (chose {selected}, answer {correct})'}\n"

            st.session_state["quiz_score"] = score
            st.session_state["quiz_results_text"] = results_text
            pct = int(score / len(st.session_state["quiz_questions"]) * 100)
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{score}/{len(st.session_state["quiz_questions"])}</div>'
                f'<div class="metric-label">Score ({pct}%)</div></div>',
                unsafe_allow_html=True,
            )
            if pct >= 80:
                st.success("🎉 Excellent! You have a strong grasp of the material.")
            elif pct >= 60:
                st.warning("📚 Good effort! Review the questions you missed.")
            else:
                st.error("⚠️ Keep studying! Check the Study Plan tab for personalized guidance.")


# ════════════════════════════════════════════════════════════
#  TAB 3 – Personalized Study Plan
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📅 Personalized Study Plan</div>', unsafe_allow_html=True)
    st.markdown("Describe your goals or weak areas and get a structured, day-by-day study plan.")

    weak_input = st.text_area(
        "What do you want to focus on or improve?",
        placeholder="e.g. I struggle with backpropagation and CNNs. I have an exam in 7 days.",
        height=120,
    )

    if st.button("🗓️ Generate Study Plan", use_container_width=True):
        if not st.session_state["api_key_set"]:
            st.warning("Set API key first.")
        elif not weak_input.strip():
            st.warning("Please describe your focus areas.")
        else:
            load_engine_and_agent()
            retriever = st.session_state["rag_engine"].get_retriever()
            with st.spinner("Creating your personalized study plan…"):
                plan = st.session_state["agent"].generate_study_plan(weak_input, retriever)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(plan)
            st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 – Weak Area Identification
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔍 Weak Area Identification</div>', unsafe_allow_html=True)
    st.markdown(
        "The AI coach will analyze your chat history and quiz results to pinpoint your knowledge gaps."
    )

    if st.button("🧠 Analyze My Learning", use_container_width=True):
        if not st.session_state["api_key_set"]:
            st.warning("Set API key first.")
        else:
            load_engine_and_agent()
            quiz_res = st.session_state.get("quiz_results_text", "")
            with st.spinner("Analyzing your learning patterns…"):
                analysis = st.session_state["agent"].identify_weak_areas(quiz_results=quiz_res)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(analysis)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("💡 Complete a quiz or chat with the tutor first, then click **Analyze My Learning** to get insights.")


# ════════════════════════════════════════════════════════════
#  TAB 5 – Practice Questions (Bonus)
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">❓ Practice Questions Generator</div>', unsafe_allow_html=True)
    st.markdown("Generate open-ended practice questions at different cognitive levels (Recall → Analysis).")

    col_x, col_y = st.columns(2)
    with col_x:
        pq_topic = st.text_input("Topic", placeholder="e.g. Transformer architecture", key="pq_topic")
    with col_y:
        num_pq = st.slider("Number of Questions", 3, 10, 5, key="num_pq")

    if st.button("🎯 Generate Practice Questions", use_container_width=True):
        if not st.session_state["api_key_set"]:
            st.warning("Set API key first.")
        elif not pq_topic.strip():
            st.warning("Enter a topic first.")
        else:
            load_engine_and_agent()
            retriever = st.session_state["rag_engine"].get_retriever()
            with st.spinner("Generating practice questions…"):
                pqs = st.session_state["agent"].generate_practice_questions(
                    pq_topic, retriever, num_questions=num_pq
                )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(pqs)
            st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.35);font-size:0.8rem;'>"
    "🎓 GenAI Learning Mentor · Built with LangChain, ChromaDB & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
