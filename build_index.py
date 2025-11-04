import os, json
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/belge.pdf"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

# 1) PDF -> metin (sayfa etiketleriyle)
pages = []
with pdfplumber.open(PDF_PATH) as pdf:
    for i, p in enumerate(pdf.pages, start=1):
        t = p.extract_text() or ""
        if t.strip():
            # Kaynak etiketi ekle
            pages.append(f"[{os.path.basename(PDF_PATH)} | sayfa {i}]\n{t}")

full_text = "\n\n".join(pages)

# 2) Parçalama (chunk)
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = splitter.split_text(full_text)

# 3) Embedding modeli (lokal, çok dilli)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# 4) FAISS index (cosine=IP + normalize kullanıyoruz)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# 5) Kaydet
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"OK: {len(chunks)} parça indekslendi.")
