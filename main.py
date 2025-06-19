import os
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from io import BytesIO
import base64
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores

# ENV VARS
CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Cool UI
st.set_page_config(page_title="🧙 D&D Sourcebook AI", layout="centered")
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

st.title("🧙 Ask your D&D Sourcebooks")


@st.cache_resource
def authenticate_drive():
    creds = None
    # Decode credentials.json from env
    with open("credentials.json", "w") as f:
        f.write(base64.b64decode(CREDENTIALS_JSON).decode())

    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service


@st.cache_data
def load_sourcebooks(service):
    results = service.files().list(q="mimeType='application/pdf'",
                                   pageSize=10,
                                   fields="files(id, name)").execute()
    files = results.get('files', [])
    texts = []

    for file in files:
        pdf_id = file['id']
        pdf_name = file['name']
        request = service.files().get_media(fileId=pdf_id)
        fh = BytesIO()
        downloader = request.execute()
        fh.write(downloader)
        fh.seek(0)

        pdf_reader = PdfReader(fh)
        for page in pdf_reader.pages:
            texts.append(page.extract_text())

    return "\n".join(texts)


@st.cache_resource
def create_vectorstore(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)


def ask_question(vectorstore, query):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vectorstore.similarity_search(query)
    return chain.run(input_documents=docs, question=query)


# App Logic
if 'vectorstore' not in st.session_state:
    with st.spinner("🔐 Authenticating with Google Drive..."):
        drive_service = authenticate_drive()

    with st.spinner("📚 Loading sourcebooks from Drive..."):
        combined_text = load_sourcebooks(drive_service)

    with st.spinner("🧠 Building vector database..."):
        vs = create_vectorstore(combined_text)
        st.session_state.vectorstore = vs

prompt = st.chat_input("Ask me something about your sourcebooks!")

if prompt:
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("🧠 Thinking..."):
        answer = ask_question(st.session_state.vectorstore, prompt)

    with st.chat_message("assistant"):
        st.markdown(f"<div class='ai-bubble'>{answer}</div>", unsafe_allow_html=True)
