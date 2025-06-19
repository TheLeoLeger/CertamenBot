import os
import json
import io
from flask import Flask, request, jsonify, render_template_string
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PyPDF2 import PdfReader
import openai

app = Flask(__name__)

# Load Google credentials JSON from env var
creds_json_str = os.getenv("GOOGLE_CREDENTIALS")
if not creds_json_str:
    raise Exception("Missing GOOGLE_CREDENTIALS environment variable")
creds_dict = json.loads(creds_json_str)
credentials = service_account.Credentials.from_service_account_info(creds_dict)

drive_service = build('drive', 'v3', credentials=credentials)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("Missing OPENAI_API_KEY environment variable")

PDF_FOLDER_ID = os.getenv("PDF_FOLDER_ID")
if not PDF_FOLDER_ID:
    raise Exception("Missing PDF_FOLDER_ID environment variable")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI PDF Chatbot</title>
<style>
  body {
    background: #121212;
    color: #eee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0; padding: 0; height: 100vh;
    display: flex; flex-direction: column;
  }
  header {
    background: #1f1f1f;
    padding: 1rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #333;
  }
  #chat {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .message {
    max-width: 70%;
    padding: 0.8rem 1rem;
    border-radius: 1rem;
    line-height: 1.4;
  }
  .user {
    background: #4a90e2;
    align-self: flex-end;
    color: white;
    border-bottom-right-radius: 0;
  }
  .bot {
    background: #2a2a2a;
    align-self: flex-start;
    color: #ddd;
    border-bottom-left-radius: 0;
  }
  form {
    display: flex;
    padding: 1rem;
    background: #1f1f1f;
    border-top: 1px solid #333;
  }
  input[type=text] {
    flex-grow: 1;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    border: none;
    outline: none;
    font-size: 1rem;
    background: #333;
    color: #eee;
  }
  button {
    margin-left: 0.75rem;
    padding: 0.75rem 1.5rem;
    border-radius: 1rem;
    border: none;
    background: #4a90e2;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.3s;
  }
  button:hover {
    background: #357ABD;
  }
  ::-webkit-scrollbar {
    width: 8px;
  }
  ::-webkit-scrollbar-thumb {
    background: #4a90e2;
    border-radius: 4px;
  }
</style>
</head>
<body>
<header>AI PDF Chatbot</header>
<div id="chat"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Ask me anything about the PDFs..." />
  <button>Send</button>
</form>
<script>
  const chat = document.getElementById('chat');
  const form = document.getElementById('form');
  const input = document.getElementById('input');

  function addMessage(text, sender) {
    const msg = document.createElement('div');
    msg.classList.add('message', sender);
    msg.textContent = text;
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    addMessage(question, 'user');
    input.value = '';
    addMessage('Thinking...', 'bot');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question})
      });
      const data = await res.json();
      chat.lastChild.textContent = data.answer || "Sorry, I didn't get that.";
    } catch {
      chat.lastChild.textContent = "Error connecting to server.";
    }
  });
</script>
</body>
</html>
"""

def list_pdfs(folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false"
    response = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return response.get('files', [])

def download_pdf(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def extract_text(pdf_filelike):
    reader = PdfReader(pdf_filelike)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)
    return "\n".join(text_chunks)

def ask_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question."})

    try:
        pdf_files = list_pdfs(PDF_FOLDER_ID)
        if not pdf_files:
            return jsonify({"answer": "No PDFs found in your Google Drive folder."})

        combined_text = ""
        for pdf in pdf_files:
            file_content = download_pdf(pdf['id'])
            combined_text += extract_text(file_content) + "\n---\n"

        prompt = f"Use the following documents to answer the question.\n\nDocuments:\n{combined_text}\n\nQuestion: {question}\nAnswer:"
        answer = ask_openai(prompt)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
