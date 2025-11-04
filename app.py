import os, json
import numpy as np
import faiss
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Yükleme
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)
index = faiss.read_index(INDEX_PATH)

EMB_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI(title="RAG-Mini (Lokal)")

def search_chunks(question, k=5):
    qv = EMB_MODEL.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    hits = [(float(D[0][i]), int(I[0][i]), CHUNKS[int(I[0][i])]) for i in range(len(I[0]))]
    return hits

def naive_answer_from_context(question, context_chunks):
    # Basit ve hızlı: en iyi 2-3 parçayı birleştirip kısa bir özet üretelim.
    # (Burada gerçek LLM yok; isterseniz Ollama/yerel LLM entegre edebilirsiniz.)
    ctx = "\n\n---\n\n".join(context_chunks)
    # Çok basit bir çıkarım: soruda geçen anahtar sözcüklere göre cümle seç.
    # (POC için yeterli. Üretimde mutlaka LLM ile özetleyin.)
    lines = []
    for c in context_chunks:
        for ln in c.split("\n"):
            if any(tok.lower() in ln.lower() for tok in question.split() if len(tok) > 3):
                lines.append(ln.strip())
    extracted = "\n".join(lines[:6]) if lines else context_chunks[0].split("\n")[0][:400]
    return extracted, ctx

@app.get("/ask")
def ask(q: str = Query(..., description="Sorunuz")):
    hits = search_chunks(q, k=6)
    top_chunks = [h[2] for h in hits]
    answer, ctx = naive_answer_from_context(q, top_chunks[:3])

    # Kaynakları çıkar (etiket cümlesi ilk satır)
    sources = []
    for ch in top_chunks[:3]:
        first_line = ch.split("\n", 1)[0].strip()
        if first_line.startswith("[") and first_line.endswith("]"):
            sources.append(first_line)

    final = f"{answer}\n\n\nKaynaklar:\n" + "\n".join(f"- {s}" for s in dict.fromkeys(sources))
    return {"question": q, "answer": final}
