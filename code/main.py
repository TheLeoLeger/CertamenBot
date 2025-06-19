import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
import openai
import numpy as np

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Load and extract text from your PDFs here (put your PDF filenames in this list)
pdf_files = ["sourcebook1.pdf", "sourcebook2.pdf"]  # change to your actual filenames

print("Loading PDFs and extracting text...")
all_text = ""
for pdf_file in pdf_files:
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"

print(f"Extracted {len(all_text)} characters of text from PDFs.")

# 2. Split text into chunks (e.g. 500 tokens each, approx 1000 chars)
CHUNK_SIZE = 1000
text_chunks = [all_text[i:i+CHUNK_SIZE] for i in range(0, len(all_text), CHUNK_SIZE)]

# 3. Create embeddings for each chunk (once at startup)
print("Generating embeddings for text chunks...")
chunk_embeddings = []
for chunk in text_chunks:
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    chunk_embeddings.append(response['data'][0]['embedding'])
chunk_embeddings = np.array(chunk_embeddings)
print(f"Generated {len(chunk_embeddings)} embeddings.")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Certamen AI Chatbot</title></head>
    <body>
        <h1>Ask me about Certamen!</h1>
        <textarea id="question" rows="3" cols="50" placeholder="Type your question here"></textarea><br/>
        <button onclick="ask()">Ask</button>
        <pre id="answer"></pre>
        <script>
            async function ask() {
                const question = document.getElementById('question').value;
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                const data = await res.json();
                document.getElementById('answer').innerText = data.answer;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")

    if not question:
        return JSONResponse({"answer": "Please ask a question."})

    # 4. Embed the question
    question_embedding_resp = openai.Embedding.create(
        input=question,
        model="text-embedding-ada-002"
    )
    question_embedding = np.array(question_embedding_resp['data'][0]['embedding'])

    # 5. Find top 3 most similar chunks
    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-3:][::-1]  # top 3

    # 6. Combine the relevant text chunks as context
    context = "\n---\n".join(text_chunks[i] for i in top_indices)

    # 7. Call OpenAI ChatCompletion with context + question
    prompt = f"You are a helpful assistant specialized in Certamen Latin quiz bowl questions.\nUse the following source material to answer:\n{context}\n\nQuestion: {question}\nAnswer:"

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer Certamen questions using provided source texts."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.0
    )

    answer = completion['choices'][0]['message']['content'].strip()

    return JSONResponse({"answer": answer})
