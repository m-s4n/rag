# check_index.py
import os, json, faiss

INDEX_DIR = "index"
idx = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("FAISS ntotal:", idx.ntotal)
print("chunks.json sayısı:", len(chunks))
print("ilk chunk önizleme:\n", chunks[0]["text"][:200], "...")
assert idx.ntotal == len(chunks), "FAISS içindeki vektör sayısı chunks ile uyuşmuyor!"
print("OK ✅")
