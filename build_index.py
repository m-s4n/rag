# -*- coding: utf-8 -*-
"""
PDF -> metin -> token-farkındalıklı chunk -> vektör (yerel/otomatik model indir)
- 1) E5 (intfloat/multilingual-e5-base) varsa yerelden kullanır
- 2) Yoksa internet VARSA otomatik indirir ve yerelde cache'ler
- 3) İnternet YOKSA TF-IDF fallback ile tamamen yerel çalışır
Çıktılar: index/faiss.index, index/chunks.json, index/config.json
"""

import os, re, json, uuid, unicodedata, sys
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tiktoken
import faiss

# ==== K O N F I G ====
PDF_PATH  = "data/belge.pdf"
OUT_DIR   = "index"
MODELS_DIR = "models"  # tüm modeller bu klasöre iner
E5_REPO_ID = "intfloat/multilingual-e5-base"
E5_LOCAL_DIR = os.path.join(MODELS_DIR, "intfloat-multilingual-e5-base")
OCR_LANG = "tur+eng"
CHUNK_TOKENS = 900
OVERLAP_TOKENS = 150

# (Opsiyonel) HF token; özel değilse boş bırak
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Windows'ta gerekiyorsa Tesseract yolu:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============ Yardımcılar: Normalizasyon/Temizlik ============
def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\u200b", "")  # zero width space
    return s

def merge_hyphenation(text: str) -> str:
    # satır sonunda tire ile bölünmüş kelimeleri birleştir
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r", "")
    # tek \n -> boşluk, \n\n paragraf kalsın
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_repeating_headers_footers(page_texts):
    first_lines, last_lines = {}, {}
    for tx in page_texts:
        lines = [l.strip() for l in tx.splitlines() if l.strip()]
        if not lines:
            continue
        first = lines[0]; last = lines[-1]
        first_lines[first] = first_lines.get(first, 0) + 1
        last_lines[last]   = last_lines.get(last, 0) + 1
    n = max(1, len(page_texts))
    headers = {k for k,v in first_lines.items() if v >= max(2, int(0.5*n))}
    footers = {k for k,v in last_lines.items()  if v >= max(2, int(0.5*n))}
    return headers, footers

def strip_headers_footers(text, headers, footers):
    lines = [l for l in text.splitlines()]
    if lines and lines[0].strip() in headers:
        lines = lines[1:]
    if lines and lines[-1].strip() in footers:
        lines = lines[:-1]
    return "\n".join(lines)

# ============ OCR fallback ============
def ocr_all_pages(pdf_path, dpi=300, lang=OCR_LANG):
    images = convert_from_path(pdf_path, dpi=dpi)
    texts = []
    for im in images:
        im = im.convert("L")
        txt = pytesseract.image_to_string(im, lang=lang)
        texts.append(normalize_unicode(txt or ""))
    return texts

# ============ PDF -> sayfa metinleri ============
def extract_pages(pdf_path):
    page_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            rot = getattr(p, "rotation", 0) or 0
            if rot in (90,180,270):
                p = p.rotate(-rot)
            t = p.extract_text() or ""
            page_texts.append(normalize_unicode(t))
    if all((not t.strip()) for t in page_texts):
        page_texts = ocr_all_pages(pdf_path)
    headers, footers = detect_repeating_headers_footers(page_texts)
    cleaned = [strip_headers_footers(t, headers, footers) for t in page_texts]
    cleaned = [merge_hyphenation(t) for t in cleaned]
    return cleaned

# ============ Token-farkındalıklı chunking ============
def chunk_texts(page_texts, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    enc = tiktoken.get_encoding("cl100k_base")
    labeled = []
    base = os.path.basename(PDF_PATH)
    for i, t in enumerate(page_texts, start=1):
        if not t.strip(): continue
        labeled.append(f"[{base} | sayfa {i}]\n{t}")
    full = "\n\n".join(labeled).strip()
    if not full: return []

    paras = full.split("\n\n")
    chunks, buf = [], []
    buf_tokens = 0

    def tok_len(s: str) -> int:
        return len(enc.encode(s))

    for para in paras:
        plen = tok_len(para)
        if plen > chunk_tokens:
            sents = re.split(r"(?<=[\.\!\?])\s+", para)
            for s in sents:
                sl = tok_len(s)
                if buf_tokens + sl <= chunk_tokens:
                    buf.append(s); buf_tokens += sl
                else:
                    if buf: chunks.append("\n".join(buf).strip())
                    if overlap_tokens > 0 and chunks:
                        tail = enc.decode(enc.encode(chunks[-1])[-overlap_tokens:])
                        buf = [tail, s]; buf_tokens = tok_len(tail) + sl
                    else:
                        buf = [s]; buf_tokens = sl
        else:
            if buf_tokens + plen <= chunk_tokens:
                buf.append(para); buf_tokens += plen
            else:
                if buf: chunks.append("\n\n".join(buf).strip())
                if overlap_tokens > 0 and chunks:
                    tail = enc.decode(enc.encode(chunks[-1])[-overlap_tokens:])
                    buf = [tail, para]; buf_tokens = tok_len(tail) + plen
                else:
                    buf = [para]; buf_tokens = plen

    if buf: chunks.append("\n\n".join(buf).strip())
    return [c for c in chunks if len(c.strip()) > 20]

# ============ Model sağlama (yerel / indir / fallback) ============
def ensure_e5_model(local_dir=E5_LOCAL_DIR, repo_id=E5_REPO_ID, hf_token=HF_TOKEN):
    """
    1) local_dir varsa: direk kullan
    2) yoksa snapshot_download ile indir (internet varsa)
    3) hata olursa None döndür (TF-IDF fallback çalışır)
    """
    if os.path.isdir(local_dir):
        return os.path.abspath(local_dir)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        # huggingface_hub yoksa kurulu olmayabilir; TF-IDF'e düşeriz
        return None

    try:
        # indir ve klasöre yaz
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token if hf_token else None,
            ignore_patterns=["*.onnx", "*.msgpack"]
        )
        return os.path.abspath(local_dir)
    except Exception as e:
        print(f"[WARN] Model indirilemedi: {e}\nTF-IDF fallback kullanılacak.")
        return None

# ============ Embedding backends ============
def embed_e5(chunks, model_dir):
    # tamamen yerel yükleme (indirildiyse)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_dir, local_files_only=True)
    payload = [f"passage: {c}" for c in chunks]
    emb = model.encode(payload, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return emb, {"backend": "e5", "model_dir": os.path.abspath(model_dir), "format": "E5 passage"}

def embed_tfidf(chunks):
    # tamamen yerel ve paketsiz alternatif (scikit gerekmez -> saf Python? pratikte scikit iyi)
    # Burada scikit-learn kullanalım (yaygın ve offline yüklenebilir)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1,2))
    X = vec.fit_transform(chunks).astype(np.float32)
    emb = X.toarray()  # küçük-orta veri için yeterli
    # normalize L2
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = (emb / norms).astype("float32")
    return emb, {"backend": "tfidf", "vocab_size": int(emb.shape[1]), "format": "tfidf-l2"}

# ============ FAISS yaz ============
def write_faiss(chunks, emb, out_dir):
    dim = emb.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    ids = np.array([uuid.uuid4().int >> 96 for _ in chunks], dtype="int64")
    index.add_with_ids(emb, ids)

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": int(i), "text": c} for i, c in zip(ids, chunks)], f, ensure_ascii=False, indent=2)
    return len(ids)

# ============ MAIN ============
if __name__ == "__main__":
    # 1) PDF -> metin
    pages = extract_pages(PDF_PATH)
    # 2) metin -> chunk
    chunks = chunk_texts(pages, CHUNK_TOKENS, OVERLAP_TOKENS)
    print(f"Chunk sayısı: {len(chunks)}")
    if not chunks:
        print("Hiç chunk üretilmedi. PDF boş olabilir."); sys.exit(1)

    # 3) model sağla: yerel / indir / tfidf fallback
    model_dir = ensure_e5_model(E5_LOCAL_DIR, E5_REPO_ID, HF_TOKEN)
    if model_dir:
        print(f"E5 modeli kullanılacak: {model_dir}")
        emb, meta = embed_e5(chunks, model_dir)
    else:
        print("E5 modeli yok/indirilemedi. TF-IDF fallback devrede.")
        emb, meta = embed_tfidf(chunks)

    # 4) FAISS yaz
    n = write_faiss(chunks, emb, OUT_DIR)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pdf": os.path.abspath(PDF_PATH),
            "embedding": meta,
            "metric": "cosine(IP+normalize)",
            "chunks": {"count": n, "token_size": CHUNK_TOKENS, "overlap": OVERLAP_TOKENS}
        }, f, ensure_ascii=False, indent=2)

    print(f"OK: {n} parça indekslendi → {OUT_DIR}")
