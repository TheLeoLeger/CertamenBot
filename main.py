import streamlit as st
import pytesseract
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import tempfile
import os
import numpy as np
import cv2
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# --- Config ---
st.set_page_config(page_title="Sourcebook QA", layout="wide")

# --- Styling ---
st.markdown("""
<style>
    .stChatMessage { background-color: #000 !important; color: #fff; padding: 0.5em; border-radius: 8px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- OCR Helper ---
def preprocess_for_ocr(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("L")
    open_cv_image = np.array(image)
    open_cv_image = cv2.resize(open_cv_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(open_cv_image, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(Image.fromarray(thresh))

# --- Extract text from scanned PDF ---
def extract_text_from_scanned_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        extracted = preprocess_for_ocr(img_bytes)
        text += extracted + "\n"
    return text

# --- Load Sourcebooks from Google Drive ---
@st.cache_data(show_spinner=True)
def load_sourcebooks(_service):
    folder_id = st.secrets["GDRIVE_FOLDER_ID"]
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = _service.files().list(q=query).execute()
    files = results.get("files", [])

    docs = []
    for file in files:
        file_id = file["id"]
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, _service.files().get_media(fileId=file_id))
        done = False
        while not done:
            status, done = downloader.next_chunk()

        text = extract_text_from_scanned_pdf(BytesIO(fh.getvalue()))
        if text.strip():
            docs.append(text)

    if not docs:
        raise ValueError("❌ No readable PDFs found. All OCR failed.")
    return docs

# --- Create Vectorstore ---
@st.cache_resource(show_spinner=True)
def create_vectorstore(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for doc in docs:
        texts.extend(text_splitter.split_text(doc))
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# --- Ask Question ---
def ask_question(vstore, query):
    docs = vstore.similarity_search(query)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    return answer

# --- Auth Google ---
def get_gdrive_service():
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return build("drive", "v3", credentials=credentials)

# --- App Start ---
st.title("📚 Sourcebook Q&A with AI")

if "vectorstore" not in st.session_state:
    with st.spinner("🔍 Loading sourcebooks and indexing..."):
        drive_service = get_gdrive_service()
        source_docs = load_sourcebooks(drive_service)
        st.session_state.vectorstore = create_vectorstore(source_docs)
        st.success("✅ Vectorstore is ready!")

query = st.text_input("Ask a question about the sourcebooks:")
if query:
    with st.spinner("🤖 Thinking..."):
        result = ask_question(st.session_state.vectorstore, query)
        st.markdown(f"**💬 Answer:** {result}")
