from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import openai
import uvicorn
import os
import io

app = FastAPI()
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    reply = f"You said: {user_message}"  # Simple echo bot for now
    return {"reply": reply}
from fastapi.responses import FileResponse
import os

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

pdf_text_storage = {}

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.post("/upload_pdf/")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    pdf_text_storage[session_id] = text
    return {"message": "PDF uploaded and text extracted", "length": len(text)}

@app.post("/chat/")
async def chat(session_id: str = Form(...), question: str = Form(...)):
    if session_id not in pdf_text_storage:
        return {"error": "No PDF uploaded for this session"}

    context_text = pdf_text_storage[session_id]

    prompt = f"""
You are a helpful assistant. Use the following document content to answer the question.
Document:
\"\"\"
{context_text}
\"\"\"
Question: {question}
Answer:"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.2,
        n=1,
        stop=None,
    )
    answer = response.choices[0].text.strip()
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
