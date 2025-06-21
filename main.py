import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # changed import

# --- ENVIRONMENT VARIABLES ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH")  # path to saved FAISS index folder

if not VECTORSTORE_PATH:
    st.error("❌ VECTORSTORE_PATH is not set.")

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="📚 PDF AI Chat", layout="wide")
st.markdown("""
    <style>
    body { background-color: #111; color: #eee; }
    .chat-user { background: #005f73; color: white; padding:8px; border-radius:8px; margin:4px 0; text-align:right; }
    .chat-ai { background: #0a9396; color:white; padding:8px; border-radius:8px; margin:4px 0; text-align:left; }
    input, button { background: #222; color: white !important; }
    </style>
""", unsafe_allow_html=True)
st.title("📚 AI Chat from Your PDFs (OCR‑enabled)")

# --- LOAD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def load_vectorstore(path):
    import os
    if not os.path.exists(path):
        st.error(f"Vectorstore path not found: {path}")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # changed here
    try:
        return FAISS.load_local(path, embeddings)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

vectorstore = load_vectorstore(VECTORSTORE_PATH)
if not vectorstore:
    st.stop()

# --- INIT QA CHAIN ---
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),  # keep ChatOpenAI as LLM
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# --- USER QUERY ---
query = st.text_input("Ask something about your sourcebooks:", key="user_input")
if query:
    with st.spinner("Thinking..."):
        res = qa({"query": query})
        ans = res["result"]
        srcs = res.get("source_documents", [])
        st.session_state.history.append((query, ans, srcs))

# --- SHOW CHAT HISTORY ---
for q, ans, sources in st.session_state.history:
    st.markdown(f"<div class='chat-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'><b>AI:</b> {ans}</div>", unsafe_allow_html=True)
    if sources:
        bullets = "\n".join(f"- {doc.metadata.get('source')}" for doc in sources)
        st.markdown(f"**Sources:**\n{bullets}")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
