# search_api.py
# Mevcut index/chunks.json, index/config.json ve index/faiss.index ile çalışan
# RAG arama servisi (ingest yok) + OPSİYONEL RERANKER (cross-encoder, tam offline).

import os
import re
import json
from typing import Dict, Any, Optional, List

import numpy as np
import faiss
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Ayarlar --------------------
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
IDX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
CH_PATH   = os.path.join(INDEX_DIR, "chunks.json")
CFG_PATH  = os.path.join(INDEX_DIR, "config.json")

# Reranker ayarları (tamamen yerel)
RERANK_ENABLE         = os.environ.get("RERANK_ENABLE", "1") == "1"   # 1=aktif, 0=pasif (genel)
RERANK_MODEL_DIR      = os.environ.get("RERANK_MODEL_DIR", "models/msmarco-MiniLM-L-6-v2-cross")
RERANK_CANDIDATES     = int(os.environ.get("RERANK_CANDIDATES", "50"))  # FAISS'ten kaç aday alınsın
RERANK_BATCH_SIZE     = int(os.environ.get("RERANK_BATCH_SIZE", "32"))  # CE batch size (CPU/GPU'na göre ayarla)

# -------------------- Global durum --------------------
app = FastAPI(title="RAG Search API (read-only) + Reranker", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

index = None
id2text: Dict[int, str] = {}
cfg: Dict[str, Any] = {}
_model = None       # E5 için SentenceTransformer
_vec = None         # TF-IDF vectorizer
_reranker = None    # CrossEncoder

# -------------------- Etiket çıkarıcı --------------------
TAG_RE = re.compile(r"\[([^\[\]]+?)\s*\|\s*sayfa\s*(\d+)\]")

def extract_tag(text: str) -> Optional[str]:
    m = TAG_RE.search(text or "")
    if not m:
        return None
    return f"{m.group(1).strip()} | sayfa {m.group(2)}"

# -------------------- Yükleme yardımcıları --------------------
def load_index_files():
    """FAISS, chunks ve config dosyalarını yükle."""
    if not (os.path.exists(IDX_PATH) and os.path.exists(CH_PATH) and os.path.exists(CFG_PATH)):
        raise FileNotFoundError("Index dosyaları eksik. Gerekli: faiss.index, chunks.json, config.json")

    faiss_idx = faiss.read_index(IDX_PATH)
    with open(CH_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    mapping = {int(c["id"]): c["text"] for c in chunks}
    return faiss_idx, mapping, config

def ensure_backend():
    """config.json'a göre embedding backend'lerini hazırla."""
    global _model, _vec
    backend = cfg.get("embedding", {}).get("backend")
    if backend == "e5":
        if _model is None:
            # tamamen offline kullan
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            model_dir = cfg["embedding"]["model_dir"]
            _model = SentenceTransformer(model_dir, local_files_only=True)
    elif backend == "tfidf":
        if _vec is None:
            from joblib import load
            vec_path = cfg["embedding"].get("vectorizer_path") or os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
            if not os.path.exists(vec_path):
                raise FileNotFoundError("TF-IDF vectorizer dosyası bulunamadı: " + vec_path)
            _vec = load(vec_path)
    else:
        raise RuntimeError(f"Bilinmeyen backend: {backend}")

def ensure_reranker():
    """Cross-encoder reranker'ı yerelden yükle (internet denemesi yapmaz)."""
    global _reranker
    if _reranker is not None or not RERANK_ENABLE:
        return

    # Eğer klasör yoksa devre dışı bırak
    if not os.path.isdir(RERANK_MODEL_DIR):
        print(f"[RERANK] '{RERANK_MODEL_DIR}' bulunamadı. Reranker devre dışı.")
        _reranker = None
        return

    # Tamamen offline kurulum
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_DIR, local_files_only=True)
        print("[RERANK] Yerel model yüklendi.")
    except Exception as e:
        print(f"[RERANK] Yüklenemedi: {e}")
        _reranker = None

def embed_query(query: str) -> np.ndarray:
    """Sorguyu E5 veya TF-IDF'e göre embedle ve float32 döndür."""
    backend = cfg["embedding"]["backend"]
    if backend == "e5":
        vec = _model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        return vec
    elif backend == "tfidf":
        v = _vec.transform([query]).astype(np.float32).toarray()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)  # cosine = IP
        return v.astype("float32")
    else:
        raise RuntimeError(f"Bilinmeyen backend: {backend}")

def startup_load():
    global index, id2text, cfg
    index, id2text, cfg = load_index_files()
    ensure_backend()
    ensure_reranker()

# -------------------- Lifecycle --------------------
@app.on_event("startup")
def on_startup():
    startup_load()

# -------------------- Endpoints --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    try:
        return {
            "ntotal": index.ntotal if index is not None else 0,
            "chunks": len(id2text),
            "backend": cfg.get("embedding", {}).get("backend"),
            "reranker": ("loaded" if _reranker is not None and RERANK_ENABLE else "disabled"),
            "reranker_model_dir": RERANK_MODEL_DIR,
            "rerank_candidates": RERANK_CANDIDATES,
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "config": cfg,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/search")
def search(
    q: str = Query(..., description="Arama sorgusu"),
    k: int = Query(5, ge=1, le=50),
    rerank: int = Query(1, description="Bu istekte reranker kullan (1) / kullanma (0)")
):
    try:
        # 1) İlk arama (geniş aday listesi)
        vec = embed_query(q)
        use_rerank = (rerank == 1) and (RERANK_ENABLE and _reranker is not None)
        cand_k = max(k, RERANK_CANDIDATES) if use_rerank else k
        cand_k = int(min(cand_k, index.ntotal))  # güvenlik

        scores, ids = index.search(vec, cand_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()

        candidates: List[Dict[str, Any]] = []
        for _id, sc in zip(ids, scores):
            if _id == -1:
                continue
            txt = id2text.get(int(_id), "")
            candidates.append({"id": int(_id), "faiss_score": float(sc), "text": txt})

        # 2) Rerank (varsa)
        rerank_used = False
        if use_rerank and len(candidates) > 0:
            pairs = [(q, c["text"]) for c in candidates]
            ce_scores = _reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE, convert_to_numpy=True)
            for c, r in zip(candidates, ce_scores):
                c["rerank_score"] = float(r)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            rerank_used = True
        else:
            candidates.sort(key=lambda x: x["faiss_score"], reverse=True)

        # 3) İlk k sonucu hazırla
        results = []
        for c in candidates[:k]:
            txt = c["text"]
            tag = extract_tag(txt)
            item = {
                "id": c["id"],
                "tag": tag,
                "preview": txt[:500].replace("\n", " "),
                "text": txt,
                "faiss_score": c["faiss_score"],
            }
            if rerank_used:
                item["rerank_score"] = c["rerank_score"]
            results.append(item)

        return {
            "query": q,
            "k": k,
            "backend": cfg["embedding"]["backend"],
            "reranker": ("enabled" if rerank_used else "disabled"),
            "results": results
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reload")
def reload_index():
    try:
        global index, id2text, cfg, _model, _vec, _reranker
        _model = None
        _vec = None
        _reranker = None
        startup_load()
        return {
            "ok": True,
            "ntotal": index.ntotal,
            "backend": cfg["embedding"]["backend"],
            "reranker": ("loaded" if _reranker is not None and RERANK_ENABLE else "disabled"),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
