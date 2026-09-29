"""KnowledgeOps AI — Streamlit frontend.

Run:
    streamlit run frontend/app.py

Requires the FastAPI backend running at http://localhost:8000.
Start it with:
    uvicorn backend.main:app --reload
"""

import os

import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="KnowledgeOps AI",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# Session state (persists across re-runs)
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []   # list of {"question", "answer", "sources", "found"}


# ============================================================
# Helpers
# ============================================================

def check_backend() -> dict | None:
    """Return /health payload or None if the backend is unreachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def ask_backend(question: str, top_k: int = 5) -> dict:
    """POST /query and return parsed JSON."""
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Status")

    health = check_backend()
    if health is None:
        st.error(f"❌ Backend unreachable at {API_URL}")
        st.caption("Start the backend: `uvicorn backend.main:app --reload`")
    else:
        st.success("✅ Backend online")
        st.metric("Indexed chunks", health["vector_store_count"])
        st.metric("LLM provider", health["llm_provider"])

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.caption("KnowledgeOps AI — Milestone 6")


# ============================================================
# Main area
# ============================================================

st.title("📚 KnowledgeOps AI")
st.caption("Ask your company knowledge base. Answers are grounded in your documents.")

# --- Input ---
with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g. How many vacation days do I get?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", use_container_width=True, type="primary")


# --- Handle submission ---
if submitted and question.strip():
    if health is None:
        st.error("Backend is offline. Start it first.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = ask_backend(question, top_k=top_k)
                st.session_state.history.insert(0, result)   # newest first
            except requests.HTTPError as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# --- Display history ---
if not st.session_state.history:
    st.info("Ask a question above, or pick one of these examples:")

    examples = [
        "How many vacation days do I get?",
        "What are the password requirements?",
        "What happens to unused vacation days?",
        "How often must I change my password?",
        "What is the maternity leave policy?",   # will trigger "not found"
    ]

    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                with st.spinner("Thinking..."):
                    try:
                        result = ask_backend(ex, top_k=top_k)
                        st.session_state.history.insert(0, result)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
