import os
import streamlit as st
from io import BytesIO

# Google Drive
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# PDF & OCR
import fitz  # PyMuPDF
import pytesseract

# LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- ENVIRONMENT VARIABLES ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
PDF_FOLDER_ID = os.getenv("PDF_FOLDER_ID")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is not set.")
if not GOOGLE_CREDS_JSON:
    st.error("❌ GOOGLE_CREDS_JSON is not set.")
if not PDF_FOLDER_ID:
    st.error("❌ PDF_FOLDER_ID is not set.")

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

# --- GOOGLE DRIVE FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def get_drive_service():
    creds_info = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

@st.cache_data(show_spinner=True)
def fetch_pdfs_from_drive(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
        fields="files(id, name)",
    ).execute()
    files = results.get("files", [])
    docs = []
    downloader = MediaIoBaseDownload
    for f in files:
        fh = BytesIO()
        req = service.files().get_media(fileId=f["id"])
        dl = downloader(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        fh.seek(0)
        docs.append((f["name"], fh.read()))
    return docs

# --- TEXT EXTRACTION WITH OCR FALLBACK ---
def extract_text(pdf_bytes: bytes) -> str:
    text = ""
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            txt = page.get_text()
            if txt and txt.strip():
                text += txt + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    # OCR fallback
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            pix = page.get_pixmap(dpi=300)
            txt = pytesseract.image_to_string(pix.tobytes("png"))
            text += txt + "\n"
        return text
    except Exception:
        return text

# --- BUILD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def build_vectorstore(pdf_list):
    docs = []
    for name, pdf in pdf_list:
        txt = extract_text(pdf)
        if not txt.strip():
            st.warning(f"⚠️ No text extracted from {name}.")
            continue
        docs.append(Document(page_content=txt, metadata={"source": name}))
    if not docs:
        st.error("No valid documents to process.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, emb)

# --- INITIALIZE ---
service = get_drive_service()
pdf_files = fetch_pdfs_from_drive(service, PDF_FOLDER_ID)
if not pdf_files:
    st.error("No PDFs found in your Drive folder.")
    st.stop()

vectorstore = build_vectorstore(pdf_files)
if not vectorstore:
    st.stop()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
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

# --- SHOW CHAT ---
for q, ans, sources in st.session_state.history:
    st.markdown(f"<div class='chat-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'><b>AI:</b> {ans}</div>", unsafe_allow_html=True)
    if sources:
        bullets = "\n".join(f"- {doc.metadata.get('source')}" for doc in sources)
        st.markdown(f"**Sources:**\n{bullets}")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
