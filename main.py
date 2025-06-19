import os
import io
import json
import asyncio
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyPDF2 import PdfReader

import openai
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env if present
openai.api_key = os.getenv("OPENAI_API_KEY")

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = "credentials.json"  # Put your Google Service Account JSON here

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Authenticate with Google Drive API
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

# Helper: List all PDF files in Drive folder (replace FOLDER_ID below)
FOLDER_ID = "your-google-drive-folder-id-here"  # <-- Replace this with your Drive folder ID

def list_pdfs_in_folder(folder_id: str) -> List[dict]:
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = drive_service.files().list(q=query, pageSize=100, fields="files(id, name)").execute()
    return results.get('files', [])

# Helper: Download a PDF file content by file ID
def download_pdf(file_id: str) -> bytes:
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

# Extract text from PDF bytes
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

# Embeddings storage (in-memory cache for demo)
embeddings_cache = []

# Create embedding vector using OpenAI
def get_embedding(text: str):
    response = openai.Embedding.create(model="text-embedding-3-large", input=text)
    return response['data'][0]['embedding']

# Simple cosine similarity
def cosine_sim(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load all PDFs from Drive, extract text, embed, and cache embeddings
def load_and_cache_embeddings():
    global embeddings_cache
    embeddings_cache = []
    pdf_files = list_pdfs_in_folder(FOLDER_ID)
    print(f"Found {len(pdf_files)} PDFs in Google Drive folder.")
    for f in pdf_files:
        print(f"Downloading {f['name']}...")
        pdf_bytes = download_pdf(f['id'])
        text = extract_text_from_pdf_bytes(pdf_bytes)
        embedding = get_embedding(text[:2000])  # limit text size for embedding
        embeddings_cache.append({"name": f["name"], "text": text, "embedding": embedding})
    print("All PDFs loaded and embedded.")

# Call once on startup (you can also do periodic refresh or manual trigger)
load_and_cache_embeddings()

# Find best matching PDF text for question
def find_best_answer(question: str) -> str:
    q_emb = get_embedding(question)
    best_score = -1
    best_text = "Sorry, I couldn't find an answer."
    for doc in embeddings_cache:
        score = cosine_sim(q_emb, doc['embedding'])
        if score > best_score:
            best_score = score
            best_text = doc['text'][:1000]  # return first 1000 chars as snippet
    return best_text

# FastAPI routes and UI

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CertamenBot Chat</title>
<style>
  body {
    margin: 0; padding: 0; background: #202123; color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  #chat {
    flex-grow: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .bubble {
    max-width: 70%; padding: 12px 18px; border-radius: 18px;
    line-height: 1.4; font-size: 16px;
  }
  .user { background: #10a37f; align-self: flex-end; border-radius: 18px 18px 0 18px; }
  .bot { background: #444654; align-self: flex-start; border-radius: 18px 18px 18px 0; }
  #input-area {
    display: flex; padding: 10px; background: #343541;
  }
  #input-area input {
    flex-grow: 1; border-radius: 18px; border: none; padding: 12px 20px;
    font-size: 16px; outline: none;
  }
  #input-area button {
    margin-left: 10px; background: #10a37f; border: none; border-radius: 18px;
    color: white; font-weight: bold; padding: 0 20px; cursor: pointer;
    font-size: 16px;
  }
</style>
</head>
<body>

<div id="chat">
  <div class="bubble bot">Hello! Ask me anything about the sourcebooks.</div>
</div>

<div id="input-area">
  <input type="text" id="userInput" placeholder="Type your question here..." autocomplete="off"/>
  <button onclick="sendQuestion()">Send</button>
</div>

<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('userInput');

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendQuestion();
  });

  async function sendQuestion() {
    let question = input.value.trim();
    if (!question) return;
    appendMessage(question, 'user');
    input.value = '';
    appendMessage('...', 'bot');

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      removeLastBotMessage();
      appendMessage(data.answer || 'Sorry, no answer.', 'bot');
      chat.scrollTop = chat.scrollHeight;
    } catch (err) {
      removeLastBotMessage();
      appendMessage('Error: ' + err.message, 'bot');
    }
  }

  function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.classList.add('bubble', sender);
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function removeLastBotMessage() {
    const bubbles = chat.querySelectorAll('.bot');
    if (bubbles.length > 0) {
      const last = bubbles[bubbles.length - 1];
      if (last.textContent === '...') last.remove();
    }
  }
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    answer = find_best_answer(question)
    return JSONResponse({"answer": answer})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
