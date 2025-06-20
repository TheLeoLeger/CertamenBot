import os
import json
from io import BytesIO
import streamlit as st

from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

# --- ENV VARS ---
CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PDF_FOLDER_ID = os.environ.get("PDF_FOLDER_ID")

# --- AUTH ---
def authenticate_drive():
    credentials_info = json.loads(CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(credentials_info)
    return build('drive', 'v3', credentials=creds)

# --- UI SETUP ---
st.set_page_config(page_title="Certamen Sourcebook AI", layout="centered")
st.markdown("""
    <style>
    .user-bubble {
        background-color: #222;
        color: #fff;
        padding: 12px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
    }
    .ai-bubble {
        background-color: #333;
        color: #fff;
        padding: 12px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Ask your Certamen Sourcebooks")

# --- OCR + LOAD PDFs ---
@st.cache_data
def load_sourcebooks(_service):
    results = _service.files().list(
        q=f"'{PDF_FOLDER_ID}' in parents and mimeType='application/pdf'",
        pageSize=50,
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])
    docs = []

    for file in files:
        pdf_id = file['id']
        name = file['name']
        try:
            request = _service.files().get_media(fileId=pdf_id)
            pdf_bytes = request.execute()

            images = convert_from_bytes(pdf_bytes)
            full_text = ""
            for img in images:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    full_text += text + "\n"

            if full_text.strip():
                docs.append(Document(page_content=full_text, metadata={"source": name}))
            else:
                st.warning(f"⚠️ OCR found no text in: {name}")

        except Exception as e:
            st.error(f"❌ Failed to load {name}: {e}")

    if not docs:
        st.error("❌ No readable PDFs found. All OCR failed.")
    else:
        st.success(f"✅ Loaded {len(docs)} documents.")
    return docs

# --- VECTORSTORE ---
@st.cache_resource
def create_vectorstore(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []

    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        for chunk in split_texts:
            chunks.append(Document(page_content=chunk, metadata=doc.metadata))

    if not chunks:
        raise ValueError("❌ No chunks found. PDF text may be empty or OCR failed.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# --- Q&A ---
def ask_question(vectorstore, query):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    chain = load_qa_chain(llm, chain_type="refine")
    docs = vectorstore.similarity_search(query, k=5)
    answer = chain.run(input_documents=docs, question=query)
    sources = "\n\n".join([
        f"📄 **{doc.metadata.get('source', 'Unknown')}**\n{doc.page_content[:300].strip()}..."
        for doc in docs
    ])
    return answer, sources

# --- SESSION SETUP ---
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# --- LOAD ONCE ---
if st.session_state.vectorstore is None:
    with st.spinner("🔐 Authenticating..."):
        drive_service = authenticate_drive()

    with st.spinner("📚 Loading sourcebooks..."):
        docs = load_sourcebooks(drive_service)

    if docs:
        with st.spinner("🧠 Building vector database..."):
            try:
                vs = create_vectorstore(docs)
                st.session_state.vectorstore = vs
            except Exception as e:
                st.error(f"❌ Vectorstore creation failed: {e}")
    else:
        st.stop()

# --- CHAT ---
prompt = st.chat_input("Ask me something about your sourcebooks!")

if prompt:
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("🤔 Thinking..."):
        answer, sources = ask_question(st.session_state.vectorstore, prompt)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='ai-bubble'><strong>📘 Answer:</strong><br>{answer}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-bubble'><strong>📎 Sources:</strong><br>{sources}</div>", unsafe_allow_html=True)
