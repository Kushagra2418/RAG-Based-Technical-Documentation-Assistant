

"""Minimal Streamlit UI (bonus).

A thin front-end over the FastAPI service for interactive Q&A. Start the API first, then
run this app.

    uvicorn app.main:app --reload          # terminal 1
    streamlit run streamlit_app.py         # terminal 2

Set ``API_URL`` in the environment if the API is not on http://localhost:8000.
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Docs Assistant", page_icon="📄", layout="centered")
st.title("📄 Technical Documentation Assistant")
st.caption("Self-corrective RAG over technical docs — LangGraph + Groq")

# One session id per browser session enables follow-up questions (conversation memory).
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: indexed documents + ingestion -------------------------------- #
with st.sidebar:
    st.header("Corpus")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
        st.metric("Sources", docs.get("total_sources", 0))
        st.metric("Chunks", docs.get("total_chunks", 0))
        for d in docs.get("documents", []):
            st.write(f"• {d['source']} ({d['chunk_count']})")
    except Exception as exc:
        st.error(f"API not reachable at {API_URL}: {exc}")

    st.divider()
    st.subheader("Ingest new documents")
    with st.form("ingest_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Files (Markdown, text, HTML, PDF)",
            type=["md", "txt", "html", "htm", "pdf"],
            accept_multiple_files=True,
        )
        url = st.text_input("Document URL (optional)")
        submitted = st.form_submit_button("Ingest")

        if submitted:
            if not uploaded_files and not url.strip():
                st.warning("Add at least one file or a URL before ingesting.")
            else:
                files_payload = [
                    ("files", (upload.name, upload.getvalue()))
                    for upload in (uploaded_files or [])
                ]
                data_payload = {"urls": url.strip()} if url.strip() else {}
                r = requests.post(
                    f"{API_URL}/ingest",
                    files=files_payload or None,
                    data=data_payload or None,
                    timeout=120,
                )
                st.success(r.json().get("message", "Done")) if r.ok else st.error(r.text)

# --- Chat history ----------------------------------------------------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    section = f" — {s['section']}" if s.get("section") else ""
                    st.markdown(f"**[{s['id']}] {s['source']}**{section}\n\n> {s['snippet']}")

# --- Chat input ------------------------------------------------------------- #
if prompt := st.chat_input("Ask about the documentation…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                ).json()
            except Exception as exc:
                resp = {"answer": f"Error: {exc}", "sources": []}

        answer = resp.get("answer", "")
        st.markdown(answer)

        meta = []
        if resp.get("query_type"):
            meta.append(f"type: {resp['query_type']}")
        if resp.get("retries"):
            meta.append(f"retries: {resp['retries']}")
        if resp.get("grounded") is not None:
            meta.append("grounded ✅" if resp["grounded"] else "grounded ⚠️")
        if resp.get("web_search_used"):
            meta.append("web search")
        if meta:
            st.caption(" · ".join(meta))

        sources = resp.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    section = f" — {s['section']}" if s.get("section") else ""
                    st.markdown(f"**[{s['id']}] {s['source']}**{section}\n\n> {s['snippet']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )