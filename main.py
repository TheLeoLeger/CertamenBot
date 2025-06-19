import os
import streamlit as st
import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from io import BytesIO
import base64
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import json
def authenticate_drive():
    credentials_json = os.getenv("GOOGLE_CREDENTIALS")
    credentials_dict = json.loads(credentials_json)
    creds = service_account.Credentials.from_service_account_info(credentials_dict)
    return build('drive', 'v3', credentials=creds)
# ENV VARS
CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
credentials_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
creds = service_account.Credentials.from_service_account_info(credentials_info)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Cool UI
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

st.title(" Ask your Certamen Sourcebooks")

 


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
