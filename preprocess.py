import os
import json
from io import BytesIO
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import fitz  # PyMuPDF
import pytesseract
import pickle
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDS")
PDF_FOLDER_ID = os.getenv("PDF_FOLDER_ID")

if not OPENAI_API_KEY or not GOOGLE_CREDS or not PDF_FOLDER_ID:
    raise EnvironmentError("Required environment variables not set.")

# --- Google Drive ---
def get_drive_service():
    creds_info = json.loads(GOOGLE_CREDS)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def fetch_pdfs_from_drive(_service, folder_id):
    results = _service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
        fie
