import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.rag import run_rag
from app.embedder import get_embeddings_batch
from app.vectorstore import ensure_collection, upsert_documents
from app.chunker import chunk_text

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    print("RAG start")
    yield

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    texts: list[str]
    metadata: list[dict] | None = None

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API server starting"}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = await run_rag(req.question, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        embeddings = await get_embeddings_batch(req.texts)
        count = upsert_documents(req.texts, embeddings, req.metadata)
        return {"message": f"{count} to save doc"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """PDF or TXT upload → Chunking → vector DB save"""
    try:
        filename = file.filename or "unknown"
        ext = filename.lower().split(".")[-1]

        if ext not in ["pdf", "txt"]:
            raise HTTPException(status_code=400, detail="only pdf or text")

        contents = await file.read()

        if ext == "pdf":
            import pdfplumber, io
            text = ""
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        else:
            for encoding in ["utf-8", "shift_jis", "euc_jp", "cp932"]:
                try:
                    text = contents.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(status_code=400, detail="fail incoding.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="not found text")

        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="not found Chunking result")

        metadata = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        embeddings = await get_embeddings_batch(chunks)
        count = upsert_documents(chunks, embeddings, metadata)

        return {
            "message": f"{filename} upload complete",
            "chunks": count,
            "text_length": len(text),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat")
def chat_ui():
    return FileResponse("static/index.html")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
