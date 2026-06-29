# -*- coding: utf-8 -*-
"""
YouTube RAG Chatbot — Streamlit App
"""

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main accent colour */
    :root { --accent: #FF4B4B; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f0f0f; }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }

    /* Chat bubbles */
    .user-bubble {
        background: #1a1a2e;
        border-left: 3px solid var(--accent);
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8e8e8;
    }
    .assistant-bubble {
        background: #16213e;
        border-left: 3px solid #4b8bff;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8e8e8;
    }
    .bubble-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
        opacity: 0.55;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ready  { background: #1a4731; color: #4ade80; }
    .badge-waiting{ background: #2d1b1b; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_title" not in st.session_state:
    st.session_state.video_title = ""

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """Accept full YouTube URLs or bare video IDs."""
    url_or_id = url_or_id.strip()
    for marker in ["v=", "youtu.be/", "shorts/"]:
        if marker in url_or_id:
            part = url_or_id.split(marker)[-1]
            return part.split("&")[0].split("?")[0]
    return url_or_id  # assume it's already a bare ID


@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str, language: str, api_key: str):
    """Fetch transcript and build FAISS vector store. Cached by video_id+lang."""
    import os
    os.environ["GOOGLE_API_KEY"] = api_key

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=[language])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks), transcript[:300]


def answer_question(question: str, vector_store, api_key: str) -> str:
    import os
    os.environ["GOOGLE_API_KEY"] = api_key

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}""",
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    response = llm.invoke(final_prompt)
    return response.content


# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 YouTube RAG Chatbot")
    st.markdown("Ask anything about any YouTube video using its transcript.")
    st.divider()

    api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        help="Your Google Gemini API key",
    )

    video_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtu.be/dQw4w9WgXcQ  or  dQw4w9WgXcQ",
    )

    language = st.selectbox(
        "Transcript language",
        options=["en", "hi", "es", "fr", "de", "pt", "ja", "ko", "zh"],
        index=0,
    )

    load_btn = st.button("⚡ Load Video", use_container_width=True, type="primary")

    if st.session_state.video_loaded:
        st.markdown(
            '<span class="status-badge badge-ready">✓ Video ready</span>',
            unsafe_allow_html=True,
        )
        if st.button("🔄 Load a different video", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.video_loaded = False
            st.session_state.chat_history = []
            st.session_state.video_title = ""
            st.rerun()
    else:
        st.markdown(
            '<span class="status-badge badge-waiting">No video loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Built with LangChain · FAISS · Gemini")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎬 YouTube RAG Chatbot")

# Load video
if load_btn:
    if not api_key:
        st.error("Please enter your Google API key in the sidebar.")
    elif not video_input:
        st.error("Please enter a YouTube URL or video ID.")
    else:
        video_id = extract_video_id(video_input)
        with st.spinner("Fetching transcript and building index…"):
            try:
                vs, num_chunks, preview = build_vector_store(video_id, language, api_key)
                st.session_state.vector_store = vs
                st.session_state.video_loaded = True
                st.session_state.video_title = video_id
                st.session_state.chat_history = []
                st.success(
                    f"✅ Loaded **{num_chunks} chunks** from video `{video_id}`."
                )
                with st.expander("Transcript preview"):
                    st.write(preview + " …")
            except TranscriptsDisabled:
                st.error("No captions available for this video.")
            except Exception as e:
                st.error(f"Error: {e}")

# Chat area
if st.session_state.video_loaded:
    st.divider()
    st.subheader("💬 Chat with the video")

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble"><div class="bubble-label">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble"><div class="bubble-label">Assistant</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask a question about the video…",
            placeholder="What topics are covered in this video?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("Thinking…"):
            try:
                reply = answer_question(
                    user_q, st.session_state.vector_store, api_key
                )
            except Exception as e:
                reply = f"⚠️ Error generating answer: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("👈 Enter your API key and a YouTube URL in the sidebar, then click **Load Video** to get started.")
