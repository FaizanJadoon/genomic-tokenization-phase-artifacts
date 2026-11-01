from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
DEST = ROOT / "models" / "nt-v2-50m"

snapshot_download(
    repo_id=MODEL,
    revision=REVISION,
    local_dir=DEST,
)
print("Model ready:", DEST)
