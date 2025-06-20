import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pytesseract
import fitz  # PyMuPDF
from io import BytesIO

# ENV variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
PDF_FOLDER_ID = os.getenv("PDF_FOLDER_ID")

# Streamlit page config
st.set_page_config(page_title="AI PDF Chat", page_icon="📄", layout="wide")

# Dark theme CSS
st.markdown(
    """
    <style>
        .css-1d391kg {background-color: #121212;}
        .css-1v3fvcr, .stTextInput>div>input, .css-ffhzg2 {color: #eee;}
        .stTextInput>div>input {background-color: #222;}
        .css-10trblm {background-color: #121212;}
        .css-ocqkz7 {background-color: #000;}
        .streamlit-expanderHeader {color: #eee;}
        .userMsg {background: #222; color: white; padding: 10px; border-radius: 8px; margin: 5px 0;}
        .botMsg {background: #333; color: white; padding: 10px; border-radius: 8px; margin: 5px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Google Drive API service
@st.cache_resource(show_spinner=False)
def get_drive_service():
    if not GOOGLE_CREDS_JSON:
        return None
    creds_dict = eval(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


@st.cache_data(show_spinner=False)
def load_pdfs_from_drive(service):
    if not service or not PDF_FOLDER_ID:
        return []
    results = (
        service.files()
        .list(
            q=f"'{PDF_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed = false",
            fields="files(id, name)",
        )
        .execute()
    )
    files = results.get("files", [])
    docs = []
    for f in files:
        file_id = f["id"]
        file_name = f["name"]
        fh = BytesIO()
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        loader = PyPDFLoader(fh)
        try:
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load {file_name}: {e}")
    return docs


def extract_text_from_pdf(file) -> str:
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        file.seek(0)

    try:
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        ocr_text = ""
        for page in pdf:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            ocr_text += pytesseract.image_to_string(img_bytes) + "\n"
        if ocr_text.strip():
            return ocr_text
    except Exception as e:
        st.error(f"OCR failed: {e}")
    return ""


@st.cache_data(show_spinner=False)
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)


def load_pdfs_from_upload(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        if not text.strip():
            st.warning(f"No text found in {uploaded_file.name}, skipping.")
            continue
        documents.append({"page_content": text, "metadata": {"source": uploaded_file.name}})
    return documents


st.title("📄 AI PDF Chat with OCR & Google Drive")

uploaded_files = st.file_uploader(
    "Upload PDFs or use the Drive folder:",
    type=["pdf"],
    accept_multiple_files=True,
)

drive_service = get_drive_service()
source_documents = []

if PDF_FOLDER_ID and drive_service:
    with st.spinner("Loading PDFs from Google Drive folder..."):
        drive_docs = load_pdfs_from_drive(drive_service)
        if drive_docs:
            source_documents.extend(drive_docs)
        else:
            st.info("No PDFs found in the Drive folder.")

if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        uploaded_docs = load_pdfs_from_upload(uploaded_files)
        source_documents.extend(uploaded_docs)

if not source_documents:
    st.warning("Upload PDFs or set a valid Google Drive folder with PDFs.")
    st.stop()

with st.spinner("Creating vector store..."):
    vectorstore = create_vectorstore(source_documents)

llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if "history" not in st.session_state:
    st.session_state.history = []

def chat(user_input):
    st.session_state.history.append(HumanMessage(content=user_input))
    response = qa_chain.run(user_input)
    st.session_state.history.append(AIMessage(content=response))

st.sidebar.header("Chat History")
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.sidebar.markdown(f"<div class='userMsg'>{msg.content}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<div class='botMsg'>{msg.content}</div>", unsafe_allow_html=True)

user_question = st.text_input("Ask a question about your PDFs:", key="input")

if user_question:
    chat(user_question)
    st.experimental_rerun()
