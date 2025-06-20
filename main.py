import os
import json
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

# --- ENV VARIABLES ---
CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PDF_FOLDER_ID = os.environ.get("PDF_FOLDER_ID")

# --- GOOGLE DRIVE AUTH ---
def authenticate_drive():
    credentials_info = json.loads(CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(credentials_info)
    service = build('drive', 'v3', credentials=creds)
    return service

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Certamen Sourcebook AI", layout="centered")
st.markdown("""
    <style>
    .user-bubble {
        background-color: #dcf8c6;
        padding: 12px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
    }
    .ai-bubble {
        background-color: #ececec;
        padding: 12px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Ask your Certamen Sourcebooks")

# --- LOAD AND PROCESS PDF SOURCEBOOKS ---
@st.cache_data
def load_sourcebooks(_service):
    results = _service.files().list(
        q=f"'{PDF_FOLDER_ID}' in parents and mimeType='application/pdf'",
        pageSize=50,
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])
    texts = []

    for file in files:
        pdf_id = file['id']
        pdf_name = file['name']
        request = _service.files().get_media(fileId=pdf_id)
        pdf_bytes = request.execute()
        fh = BytesIO(pdf_bytes)
        pdf_reader = PdfReader(fh)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"source": pdf_name})
                texts.append(doc)

    return texts
    st.write(f"Loaded {len(texts)} documents")
if not texts:
    st.warning("No text extracted from PDFs")
# --- VECTORSTORE CREATION ---
@st.cache_resource
def create_vectorstore(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc.page_content):
            chunks.append(Document(page_content=chunk, metadata=doc.metadata))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# --- QUESTION ANSWERING LOGIC ---
def ask_question(vectorstore, query):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    chain = load_qa_chain(llm, chain_type="refine")  # or "map_reduce" for broader answers
    docs = vectorstore.similarity_search(query, k=5)
    answer = chain.run(input_documents=docs, question=query)

    # Format citations from metadata
    sources = "\n\n".join([
        f"📄 **{doc.metadata.get('source', 'Unknown Source')}**\n{doc.page_content[:300].strip()}..."
        for doc in docs
    ])

    return answer, sources

# --- MAIN APP LOGIC ---
if 'vectorstore' not in st.session_state:
    with st.spinner("🔐 Authenticating with Google Drive..."):
        drive_service = authenticate_drive()

    with st.spinner("📚 Loading sourcebooks from Drive..."):
        source_documents = load_sourcebooks(drive_service)

    with st.spinner("🧠 Building vector database..."):
        vs = create_vectorstore(source_documents)
        st.session_state.vectorstore = vs

prompt = st.chat_input("Ask me something about your sourcebooks!")

if prompt:
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("🧠 Thinking..."):
        answer, sources = ask_question(st.session_state.vectorstore, prompt)

    with st.chat_message("assistant"):
        st.markdown("""
            <div class='ai-bubble'>
            <strong>📘 Answer:</strong><br>
        """, unsafe_allow_html=True)
        st.markdown(answer, unsafe_allow_html=True)

        st.markdown("""
            <br><strong>📎 Sources I used:</strong><br>
        """, unsafe_allow_html=True)
        st.markdown(sources, unsafe_allow_html=True)
