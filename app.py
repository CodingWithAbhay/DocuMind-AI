"""DocuMind AI - chat with your documents (Streamlit + Gemini embeddings + Groq)."""

from __future__ import annotations

import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from extractor import SUPPORTED_EXTENSIONS, UnsupportedDocument, extract_text
from rag import (
    ConfigError,
    answer_question,
    chunk_text,
    embed_document,
    embed_query,
    top_matches,
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

st.set_page_config(page_title="DocuMind AI", page_icon="📜", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Space+Grotesk:wght@400;500;600&display=swap');

      :root {
        --paper:     #f6f1e7;
        --cream:     #fffaf0;
        --ink:       #2c2620;
        --muted:     #7a6d5d;
        --line:      #ddd0b8;
        --ochre:     #a66f21;
        --pine:      #356555;
        --soft-pine: #e3ede7;
      }

      /* Base layout & typography */
      .stApp { background: var(--paper); color: var(--ink); }
      html, body, p, label, input, textarea, button, .stMarkdown { font-family: 'Space Grotesk', sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink); letter-spacing: 0 !important; }

      /* Hide Streamlit default UI overlays */
      [data-testid="stHeader"],
      header[data-testid="stHeader"],
      [data-testid="stToolbar"],
      .stAppDeployButton,
      [data-testid="stDecoration"],
      #MainMenu,
      footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
      }

      /* Screen width layout */
      .stMainBlockContainer {
        max-width: 1550px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
      }

      /* Brand Header */
      .dm-brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding-bottom: .5rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: .75rem;
      }
      .dm-mark {
        width: 34px; height: 34px;
        display: grid; place-items: center;
        border-radius: 6px;
        background: var(--ink); color: var(--cream);
        font: 600 18px 'Fraunces', serif;
      }
      .dm-title { font: 600 1.25rem 'Fraunces', serif; line-height: 1.05; }
      .dm-sub   { color: var(--muted); font-size: .75rem; margin-top: .15rem; }

      /* Left Panel Container */
      [data-testid="stHorizontalBlock"] > div:first-child {
        background: #faf6ee;
        border: 1px solid #e2d6c1;
        border-radius: 12px;
        padding: 1.25rem 1.4rem !important;
        box-shadow: 0 2px 10px rgba(90,65,30,.05);
        min-height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }

      .dm-kicker {
        color: var(--ochre);
        font-size: .7rem;
        font-weight: 600;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .2rem;
      }
      .dm-upload-intro h2 { font-size: 1.25rem; margin: 0 0 .2rem; color: var(--ink); }
      .dm-upload-intro p  { color: var(--muted); font-size: .8rem; line-height: 1.4; margin: 0; }

      .dm-limit {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #efe6d5;
        border: 1px solid var(--line);
        color: #674c20;
        padding: .45rem .8rem;
        border-radius: 6px;
        font-size: .78rem;
        margin: .5rem 0;
      }

      /* Drag & Drop Area */
      [data-testid="stFileUploader"] {
        background: var(--cream) !important;
        border: 1.5px dashed #c8b691 !important;
        border-radius: 10px !important;
        padding: .5rem .6rem !important;
        margin: .35rem 0 0.15rem 0 !important;
        transition: border-color 0.2s ease !important;
      }
      [data-testid="stFileUploader"]:hover {
        border-color: var(--ochre) !important;
      }

      [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: 0 !important;
        min-height: 65px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: .5rem !important;
        padding: .2rem .4rem !important;
      }

      [data-testid="stFileUploaderDropzone"] > button {
        background: var(--ink) !important;
        color: var(--cream) !important;
        border: 1px solid var(--ink) !important;
        border-radius: 6px !important;
        min-width: 95px !important;
        height: 32px !important;
        padding: .2rem .75rem !important;
        font-size: .76rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
      }
      [data-testid="stFileUploaderDropzone"] > button:hover {
        background: var(--ochre) !important;
        border-color: var(--ochre) !important;
        color: #ffffff !important;
      }

      [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        align-items: center !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] span,
      [data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "Drag & drop files or browse" !important;
        color: var(--muted) !important;
        font-size: .78rem !important;
        font-weight: 500 !important;
      }

      /* Supported Documents Badges */
      .dm-supported-docs {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: .35rem;
        margin: .3rem 0 .5rem;
      }
      .dm-supported-label {
        font-size: .7rem;
        color: var(--muted);
        font-weight: 500;
        margin-right: .2rem;
      }
      .dm-doc-badge {
        background: #efe6d5;
        border: 1px solid #dcd0ba;
        color: var(--ink);
        padding: .1rem .4rem;
        border-radius: 4px;
        font-size: .65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .03em;
      }

      /* File Items */
      [data-testid="stFileUploaderFile"] {
        background: #f0e7d5 !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        padding: .35rem .6rem !important;
        margin: .2rem 0 !important;
      }

      /* Status Badge & Button */
      .dm-status {
        background: var(--cream);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .55rem .8rem;
        margin: .4rem 0 !important;
      }
      .dm-status-head { display: flex; justify-content: space-between; align-items: center; font-size: .78rem; }
      .dm-status-name { font-weight: 600; color: var(--ink); }
      .dm-ready  { color: var(--pine); font-weight: 600; font-size: .74rem; }
      .dm-track  { height: 3px; background: #e8dfce; border-radius: 10px; overflow: hidden; margin-top: .35rem; }
      .dm-track span { display: block; width: 100%; height: 100%; background: var(--pine); }

      /* Clear button */
      [data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
        background: var(--ink) !important;
        color: var(--cream) !important;
        border: 0 !important;
        border-radius: 7px !important;
        width: 100% !important;
        font-size: .82rem !important;
        font-weight: 500 !important;
        padding: .45rem .8rem !important;
        margin-top: .4rem !important;
      }
      [data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover:not([disabled]) {
        background: var(--ochre) !important;
        color: #ffffff !important;
      }

      /* Services Footer */
      .dm-powered {
        color: var(--muted);
        font-size: .72rem;
        line-height: 1.45;
        padding-top: .6rem !important;
        border-top: 1px solid var(--line);
        margin-top: .6rem !important;
      }

      /* Right Chat Panel Layout */
      .dm-chat-head { display: flex; align-items: end; justify-content: space-between; margin-bottom: .5rem; }
      .dm-chat-head h1 { font-size: 1.5rem; margin: 0; line-height: 1.15; }
      .dm-chat-head p  { color: var(--muted); font-size: .8rem; margin: .15rem 0 0; }
      .dm-source-count { color: var(--pine); background: var(--soft-pine); padding: .25rem .55rem; border-radius: 999px; font-size: .7rem; font-weight: 600; }

      .dm-empty {
        min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 2rem 1.5rem;
        margin-top: .3rem;
      }
      .dm-empty-icon {
        width: 44px; height: 44px;
        display: grid; place-items: center;
        border: 1px solid var(--line);
        border-radius: 50%;
        color: var(--ochre);
        font: 600 1.25rem 'Fraunces', serif;
      }
      .dm-empty h3 { font-size: 1.2rem; margin: .75rem 0 .25rem; }
      .dm-empty p  { max-width: 400px; color: var(--muted); font-size: .82rem; line-height: 1.45; margin: 0; }

      .dm-ready-state {
        background: var(--cream);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.25rem 1.1rem;
        margin: .25rem 0 .5rem;
        text-align: center;
      }
      .dm-ready-icon {
        width: 34px; height: 34px;
        display: grid; place-items: center;
        border-radius: 50%;
        background: var(--soft-pine);
        color: var(--pine);
        margin: 0 auto .4rem;
      }

      /* Chat messages & Input */
      [data-testid="stChatMessage"] { background: transparent; padding: .3rem 0; }
      [data-testid="stChatMessageContent"] { background: var(--cream); border: 1px solid #eadfca; border-radius: 8px; padding: .6rem .85rem; font-size: .88rem; }

      [data-testid="stChatInput"] {
        background: var(--cream);
        border: 1.5px solid var(--line);
        border-radius: 8px;
        margin-top: .35rem;
      }
      [data-testid="stChatInput"]:focus-within { border-color: var(--ochre) !important; }

      /* Responsive layout handling */
      @media (max-width: 900px) {
        .stMainBlockContainer { padding: .4rem .8rem !important; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
        [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
        .dm-empty { min-height: 240px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dm-brand"><div class="dm-mark">D</div><div>'
    '<div class="dm-title">DocuMind AI</div>'
    '<div class="dm-sub">Chat With Your Documents.</div></div></div>',
    unsafe_allow_html=True,
)

for key, default in {
    "chunks": None,
    "matrix": None,
    "doc_names": [],
    "doc_name": None,
    "messages": [],
    "uploader_key": 0,
}.items():
    st.session_state.setdefault(key, default)

upload_panel, chat_panel = st.columns([0.34, 0.66], gap="large")

with upload_panel:
    st.markdown(
        f'<div class="dm-kicker">Source documents</div>'
        f'<div class="dm-upload-intro"><h2>Bring your documents</h2>'
        f'<p>Drop in one or multiple files and DocuMind will turn them into searchable passages.</p></div>'
        f'<div class="dm-limit"><span>Upload allowance</span><strong>Up to {MAX_UPLOAD_MB} MB / file</strong></div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=SUPPORTED_EXTENSIONS,
        help=f"The maximum upload size is {MAX_UPLOAD_MB} MB per file.",
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # UI badges displaying only the specified formats
    st.markdown(
        '<div class="dm-supported-docs">'
        '<span class="dm-supported-label">Supported formats:</span>'
        '<span class="dm-doc-badge">.pdf</span>'
        '<span class="dm-doc-badge">.docx</span>'
        '<span class="dm-doc-badge">.pptx</span>'
        '<span class="dm-doc-badge">.xlsx</span>'
        '<span class="dm-doc-badge">.xls</span>'
        '<span class="dm-doc-badge">.txt</span>'
        '<span class="dm-doc-badge">.md</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    current_files = uploaded_files if uploaded_files is not None else []
    current_names = [f.name for f in current_files]

    if current_names and current_names != (st.session_state.get("doc_names") or []):
        oversized = [
            f.name for f in current_files if len(f.getvalue()) / (1024 * 1024) > MAX_UPLOAD_MB
        ]
        if oversized:
            st.error(f"These file(s) exceed {MAX_UPLOAD_MB} MB: {', '.join(oversized)}")
        else:
            try:
                num = len(current_files)
                with st.status(
                    f"Processing {num} document{'s' if num > 1 else ''}…", expanded=True
                ) as status:
                    all_chunks: list[str] = []
                    for idx, f in enumerate(current_files, start=1):
                        status.update(label=f"Reading ({idx}/{num}): {f.name}…")
                        data = f.getvalue()
                        text = extract_text(f.name, data)
                        doc_chunks = chunk_text(text)
                        if num > 1:
                            doc_chunks = [f"[{f.name}] {p}" for p in doc_chunks]
                        all_chunks.extend(doc_chunks)

                    if len(all_chunks) > 800:
                        all_chunks = all_chunks[:800]

                    status.update(label=f"Indexing {len(all_chunks)} passages with Gemini…")
                    matrix = embed_document(all_chunks)
                    status.update(
                        label=f"{num} document{'s' if num > 1 else ''} ready ✓",
                        state="complete",
                        expanded=False,
                    )

                st.session_state.chunks = all_chunks
                st.session_state.matrix = matrix
                st.session_state.doc_names = current_names
                st.session_state.doc_name = ", ".join(current_names)
                st.session_state.messages = []
                st.rerun()
            except (UnsupportedDocument, ConfigError) as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not process documents: {exc}")
    elif not current_names and st.session_state.get("doc_names"):
        st.session_state.chunks = None
        st.session_state.matrix = None
        st.session_state.doc_names = []
        st.session_state.doc_name = None
        st.session_state.messages = []
        st.rerun()

    if st.session_state.chunks and st.session_state.doc_names:
        num_docs = len(st.session_state.doc_names)
        title_text = f"{num_docs} document{'s' if num_docs > 1 else ''}"
        st.markdown(
            f'<div class="dm-status">'
            f'<div class="dm-status-head">'
            f'<span class="dm-status-name">✦ {title_text} indexed</span>'
            f'<span class="dm-ready">Ready · {len(st.session_state.chunks)} chunks</span>'
            f'</div>'
            f'<div class="dm-track"><span></span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    has_active_docs = bool(
        st.session_state.doc_names
        or st.session_state.doc_name
        or st.session_state.chunks
        or st.session_state.messages
    )
    if st.button("Clear documents & chat", disabled=not has_active_docs):
        for key in ("chunks", "matrix", "doc_name", "messages"):
            st.session_state[key] = [] if key == "messages" else None
        st.session_state.doc_names = []
        st.session_state.uploader_key += 1
        st.rerun()

    google_ready = bool(os.getenv("GOOGLE_API_KEY"))
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    service_icon = "✦" if google_ready and groq_ready else "⚠"
    service_label = (
        "<strong>Services connected</strong>"
        if google_ready and groq_ready
        else "<strong>Add API keys in .env</strong>"
    )
    st.markdown(
        f'<div class="dm-powered">{service_icon} {service_label}<br>'
        f'<span>Gemini embeddings · Groq answers</span></div>',
        unsafe_allow_html=True,
    )

with chat_panel:
    num_docs = (
        len(st.session_state.doc_names)
        if st.session_state.get("doc_names")
        else (1 if st.session_state.chunks else 0)
    )
    source_count = (
        f"{num_docs} source{'s' if num_docs != 1 else ''}"
        if st.session_state.chunks
        else "No source"
    )
    st.markdown(
        f'<div class="dm-chat-head"><div><h1>Ask your documents</h1>'
        '<p>Answers stay grounded in the passages you uploaded.</p></div>'
        f'<span class="dm-source-count">{source_count}</span></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chunks:
        st.markdown(
            '<div class="dm-empty"><div class="dm-empty-icon">D</div>'
            '<h3>Your reading desk is empty</h3>'
            '<p>Upload one or more documents on the left. Once indexed, you can ask questions '
            'and inspect the exact passages behind every answer.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        chat_box = st.container(height=380, border=False)
        with chat_box:
            if not st.session_state.messages:
                doc_count = len(st.session_state.doc_names) if st.session_state.doc_names else 1
                doc_label = f"{doc_count} document{'s' if doc_count > 1 else ''}"
                st.markdown(
                    f'<div class="dm-ready-state">'
                    f'<div class="dm-ready-icon">✦</div>'
                    f'<h3>{doc_label.capitalize()} ready to explore</h3>'
                    f'<p>Passages are indexed and grounded. Ask any question below to begin chatting with your documents.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                for message in st.session_state.messages:
                    avatar = "👤" if message["role"] == "user" else "🤖"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])

        question = st.chat_input("Ask something about your documents…")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with chat_box:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(question)

                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        with st.spinner("Searching your documents…"):
                            query_vector = embed_query(question)
                            matches = top_matches(
                                query_vector, st.session_state.matrix, st.session_state.chunks
                            )
                            history = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                            raw_answer = answer_question(question, matches, history)
                            answer = re.sub(r"\[Chunk\s*\d+\]", "", raw_answer)
                            answer = re.sub(r"\(Chunk\s*\d+\)", "", answer)
                            answer = re.sub(r"[ \t]+", " ", answer).strip()

                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    except ConfigError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"The assistant could not answer: {exc}")

            st.rerun()
