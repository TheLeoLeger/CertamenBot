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
