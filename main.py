import os
import json
import tempfile
import streamlit as st
from io import BytesIO

# Google API
from google.oauth2 import service_account
from googleapiclient.discovery import build

# PDF & OCR
import fitz  # PyMuPDF
import pytesseract

# LangChain and OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


# === ENVIRONMENT VARIABLES / SECRETS ===

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = st.secrets.get("GOOGLE_CREDENTIALS_JSON") or os.getenv("GOOGLE_CREDENTIALS_JSON")
PDF_FOLDER_ID = st.secrets.get("PDF_FOLDER_ID") or os.getenv("PDF_FOLDER_ID")


# === GOOGLE DRIVE SERVICE SETUP ===

def get_drive_service():
    if not GOOGLE_CREDENTIALS_JSON:
        st.error("Google credentials not found!")
        return None
    creds_info = json.loads(GOOGLE_CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)
    return service


def list_pdfs_in_folder(service, folder_id):
    """List PDF files in a Google Drive folder."""
    results = []
    page_token = None
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            )
            .execute()
        )
        files = response.get('files', [])
        results.extend(files)
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return results


def download_file(service, file_id):
    """Download file bytes from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    from googleapiclient.http import MediaIoBaseDownload
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()


# === PDF TEXT EXTRACTION WITH OCR FALLBACK ===

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    Try text extraction first; if empty, do OCR per page.
    """
    text = ""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text = "\n".join([page.page_content for page in pages])
    except Exception as e:
        st.warning(f"Standard PDF extraction failed: {e}")

    # If text empty or too short, fallback to OCR
    if len(text.strip()) < 20:
        st.info("Text extraction empty or too short. Running OCR fallback...")
        try:
            doc = fitz.open(file_path)
            ocr_text = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes(output="png")
                text_img = pytesseract.image_to_string(BytesIO(img_bytes))
                ocr_text.append(text_img)
            text = "\n".join(ocr_text)
        except Exception as e:
            st.error(f"OCR fallback failed: {e}")
            text = ""

    if len(text.strip()) == 0:
        st.error("No text extracted from PDF even after OCR fallback.")
    return text


# === DOCUMENT PROCESSING AND VECTOR STORE CREATION ===

@st.cache_data(show_spinner=True)
def process_and_create_vectorstore(pdf_bytes_list):
    """
    Receives list of tuples [(filename, bytes)], extracts text,
    splits to chunks, creates FAISS vectorstore and returns it.
    """
    from langchain.schema import Document

    documents = []

    for filename, pdf_bytes in pdf_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            text = extract_text_from_pdf(tmp_file.name)
            if not text:
                continue
            # Create Document with metadata filename
            documents.append(Document(page_content=text, metadata={"source": filename}))

    if not documents:
        st.error("No documents loaded to create vectorstore.")
        return None

    # Split long texts into chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# === SETUP RETRIEVAL-BASED QA CHAIN ===

def get_qa_chain(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain


# === STREAMLIT UI ===

def main():
    st.set_page_config(page_title="AI PDF Chat — OCR + LangChain", layout="wide", initial_sidebar_state="expanded")

    # Dark theme styling
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stTextInput>div>div>input {
            background-color: #1f1f1f !important;
            color: #fff !important;
        }
        .stButton>button {
            background-color: #333 !important;
            color: #fff !important;
        }
        .stMarkdown, .css-10trblm, .css-1d391kg {
            color: #e0e0e0 !important;
        }
        .chat-message {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
        }
        .chat-user {
            background-color: #005f73;
            color: white;
            text-align: right;
        }
        .chat-ai {
            background-color: #0a9396;
            color: white;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📚 AI PDF Chat with OCR & LangChain")

    # Sidebar: Upload PDFs (multiple)
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDFs (supports scanned PDFs via OCR)", type=["pdf"], accept_multiple_files=True
    )

    # Load PDFs from Google Drive folder if PDF_FOLDER_ID is set
    pdf_bytes_list = []
    if PDF_FOLDER_ID:
        st.sidebar.info(f"Loading PDFs from Google Drive folder ID: {PDF_FOLDER_ID}")
        service = get_drive_service()
        if service:
            try:
                pdf_files = list_pdfs_in_folder(service, PDF_FOLDER_ID)
                for f in pdf_files:
                    file_bytes = download_file(service, f['id'])
                    pdf_bytes_list.append((f['name'], file_bytes))
                st.sidebar.success(f"Loaded {len(pdf_files)} PDFs from Google Drive folder")
            except Exception as e:
                st.sidebar.error(f"Error loading PDFs from Google Drive: {e}")
        else:
            st.sidebar.error("Google Drive service not available. Check your credentials.")

    # Add uploaded PDFs to list
    if uploaded_files:
        for file in uploaded_files:
            pdf_bytes_list.append((file.name, file.read()))

    if not pdf_bytes_list:
        st.warning("Please upload PDFs or set a valid PDF_FOLDER_ID to load documents.")
        return

    st.info(f"Processing {len(pdf_bytes_list)} PDF(s)... This can take some seconds.")

    vectorstore = process_and_create_vectorstore(pdf_bytes_list)
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.success("Vectorstore created. You can now chat with your documents!")
    else:
        return

    qa = get_qa_chain(st.session_state.vectorstore)

    # Session state init for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    query = st.text_input("Ask anything about your documents:", key="input")
    if query:
        with st.spinner("Thinking..."):
            result = qa({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])

            st.session_state.chat_history.append({"question": query, "answer": answer, "sources": sources})

    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(
            f'<div class="chat-message chat-user">**You:** {chat["question"]}</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message chat-ai">**AI:** {chat["answer"]}</
