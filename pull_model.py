from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="models/msmarco-MiniLM-L-6-v2-cross",
    local_dir_use_symlinks=False
)
