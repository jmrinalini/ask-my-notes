"""
Phase 9: Streamlit UI.

Everything up to now has been scripts you run from a terminal. This phase
wraps the same underlying pipeline (Phase 1-5, 7) in a simple web
interface: upload a PDF, pick a mode, ask questions, see cited answers -
no terminal required.

Nothing about the CORE RAG logic changes here. This file is mostly
"plumbing": take user input from web widgets instead of hardcoded
strings, call the same functions we already built, display results nicely.
That's an intentional and common pattern - keep your core logic
(chunking, embedding, retrieval, generation) separate from your interface,
so you can swap interfaces (CLI, web, API) without touching the logic.

Run with:  streamlit run src/phase9_app.py
"""

import os
import tempfile
import hashlib

import streamlit as st

from phase1_ingest import extract_text_from_pdf, chunk_text
from phase3_vector_store import build_vector_store, search
from phase4_generate import ask_with_context, MODE_INSTRUCTIONS
from phase5_citations import ask_with_citations


st.set_page_config(page_title="Ask My Notes", page_icon="📄")
st.title("📄 Ask My Notes")
st.caption("Upload a document, then ask questions, get things explained, or summarized — grounded in and cited from your own file.")


def get_collection_name(file_bytes: bytes) -> str:
    """
    Build a stable, unique collection name from the file's content, so
    re-uploading the SAME file reuses the existing vectors instead of
    re-embedding everything from scratch every time.
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()[:10]
    return f"doc_{file_hash}"


@st.cache_resource(show_spinner="Reading and indexing your document...")
def process_pdf(file_bytes: bytes, collection_name: str):
    """
    Save the uploaded PDF temporarily, run it through the full ingestion
    pipeline (Phase 1-3), and return a ready-to-query collection.
    st.cache_resource means this only runs ONCE per unique file - if the
    person asks multiple questions, we don't redo this every time.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    collection = build_vector_store(tmp_path, collection_name=collection_name)
    os.unlink(tmp_path)  # clean up the temp file, we don't need it anymore

    return collection


# --- Sidebar: file upload and mode selection ---
with st.sidebar:
    st.header("1. Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    st.header("2. Choose a mode")
    mode = st.radio(
        "What do you want to do?",
        options=["qa", "simplify", "summarize"],
        format_func=lambda m: {
            "qa": "Ask a question",
            "simplify": "Simplify a topic",
            "summarize": "Summarize a topic",
        }[m],
    )

    show_citations = st.checkbox(
        "Show citations (Q&A mode only)",
        value=True,
        help="Shows which part of your document each claim came from."
    )


# --- Main area: question input and answer ---
if uploaded_file is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    file_bytes = uploaded_file.read()
    collection_name = get_collection_name(file_bytes)
    collection = process_pdf(file_bytes, collection_name)

    st.success(f"'{uploaded_file.name}' is ready. Ask away!")

    placeholder_text = {
        "qa": "e.g. What were the final performance metrics?",
        "simplify": "e.g. the Grey Wolf Optimizer algorithm",
        "summarize": "e.g. the results and analysis section",
    }[mode]

    user_input = st.text_input("Your question or topic:", placeholder=placeholder_text)

    if st.button("Submit") and user_input:
        with st.spinner("Thinking..."):
            if mode == "qa" and show_citations:
                answer, sources = ask_with_citations(collection, user_input, top_k=3)
            else:
                answer = ask_with_context(collection, user_input, mode=mode, top_k=3)
                sources = None

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("### Sources (to verify the citations above)")
            for i, source in enumerate(sources):
                with st.expander(f"Source {i + 1}"):
                    st.write(source)