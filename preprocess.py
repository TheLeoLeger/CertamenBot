import os
import json
from io import BytesIO

import fitz  # PyMuPDF
import pytesseract

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --- ENVIRONMENT VARIABLES ---
GOOGLE_CREDS = os.getenv("GOOGLE_CREDS")
PDF_FOLDER_ID = os.getenv("PDF_FOLDER_ID")
VECTORSTORE_SAVE_PATH = os.getenv("VECTORSTORE_PATH", "faiss_index")  # default save folder

if not GOOGLE_CREDS or not PDF_FOLDER_ID:
    raise ValueError("Please set GOOGLE_CREDS and PDF_FOLDER_ID environment variables.")

# --- SETUP GOOGLE DRIVE SERVICE ---
def get_drive_service():
    creds_info = json.loads(GOOGLE_CREDS)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

# --- FETCH PDFs FROM DRIVE ---
def fetch_pdfs_from_drive(_service, folder_id):
    print("Fetching PDFs from Google Drive folder...")
    results = _service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
        fields="files(id, name)",
    ).execute()
    files = results.get("files", [])
    if not files:
        print("No PDF files found in the folder.")
        return []
    docs = []
    downloader = MediaIoBaseDownload
    for f in files:
        print(f"Downloading {f['name']}...")
        fh = BytesIO()
        req = _service.files().get_media(fileId=f["id"])
        dl = downloader(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        fh.seek(0)
        docs.append((f["name"], fh.read()))
    print(f"Downloaded {len(docs)} PDFs.")
    return docs

# --- EXTRACT TEXT WITH OCR FALLBACK ---
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
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")

    # OCR fallback
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            pix = page.get_pixmap(dpi=300)
            txt = pytesseract.image_to_string(pix.tobytes("png"))
            text += txt + "\n"
        return text
    except Exception as e:
        print(f"Error during OCR fallback: {e}")
        return text

# --- MAIN PREPROCESS FUNCTION ---
def main():
    service = get_drive_service()
    pdf_files = fetch_pdfs_from_drive(service, PDF_FOLDER_ID)
    if not pdf_files:
        print("No PDFs found, exiting.")
        return

    documents = []
    for name, pdf in pdf_files:
        print(f"Extracting text from {name}...")
        text = extract_text(pdf)
        if not text.strip():
            print(f"Warning: No text extracted from {name}. Skipping.")
            continue
        documents.append(Document(page_content=text, metadata={"source": name}))

    if not documents:
        print("No valid documents after text extraction. Exiting.")
        return

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving vectorstore locally to '{VECTORSTORE_SAVE_PATH}'...")
    vectorstore.save_local(VECTORSTORE_SAVE_PATH)
    print("Preprocessing complete.")
    
if __name__ == "__main__":
    main()
