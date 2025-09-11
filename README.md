# Wiki‑in‑a‑Box (Offline Wikipedia - Hybrid no‑index RAG) - Powered by gpt-oss:20b

Runs a local LAN hotspot that answers questions grounded on a local Wikipedia ZIM, with clickable citations. Uses Ollama for gpt‑oss:20b.

## Quick start

1) Start containers
- `docker compose up -d nginx kiwix api`

2) Add the ZIM
- Easiest: `make prefetch-zim` (downloads to `./data/zims/enwiki.zim`).
- Or manually place a no-images ZIM at `./data/zims/enwiki.zim` (host). The API is ready immediately — no pre‑index build.
- If you place the ZIM after the containers are up, run: `docker compose restart api` so it picks up the file.

Get the ZIM (manual mirrors)
- No-images English (Aug 2025):
  - Mirror 1: https://gemmei.ftp.acc.umu.se/mirror/kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-08.zim
  - Mirror 2: https://ftp.fau.de/kiwix/zim/wikipedia/wikipedia_en_all_nopic_2025-08.zim
- One-liner (manual):
  - `mkdir -p ./data/zims && curl -L -o ./data/zims/enwiki.zim https://gemmei.ftp.acc.umu.se/mirror/kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-08.zim`

3) Run Ollama locally (host)
- The API talks to the host Ollama at `http://host.docker.internal:11434/v1`.
- Pull the model once on the host: `ollama pull gpt-oss:20b`.
- See detailed steps below in “Running Ollama locally”.

4) Use it
- Open `http://localhost` and ask a question.

Frontend (dev, optional)
- Serve the static frontend without Docker (auto‑detects API):
  - `make serve-frontend` → open `http://localhost:5173`
  - If Docker stack is running, the page uses `/api` (Nginx). Otherwise it falls back to `http://localhost:8000/api`.

## Offline Mode (No WAN)

This project can run fully offline after a one‑time embedding model prefetch. There is no long pre‑index build step — the API starts instantly.

What “offline” means here
- The API reads the ZIM locally and recalls candidate articles using libzim’s built‑in full‑text index.
- It then does semantic re‑ranking of text chunks using a local SentenceTransformers model.
- The only network dependency is downloading the model weights once; after prefetch, all queries are fully offline.

Repository defaults already set
- `docker-compose.yml` mounts `./data/hf` → `/root/.cache/huggingface:ro` in the API container.
- It sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so model loading never hits the network.

One‑time model prefetch (Docker‑only, recommended) - prefetch once and package it
```
mkdir -p ./data/hf
docker run --rm -v "$PWD/data/hf:/root/.cache/huggingface" python:3.12-slim bash -lc '
  set -e
  pip install --no-cache-dir huggingface_hub
  python - << "PY"
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-en-v1.5")
print("cached")
PY
'
```

Verify cache is populated
```
ls -lah ./data/hf
```

Run (offline)
- Ensure your ZIM is present at `./data/zims/enwiki.zim`.
- Start API stack: `docker compose up -d --build api nginx kiwix`
- Tail logs: `docker compose logs -f api` — you should see `[search] using libzim search over /data/zims/enwiki.zim`.

Notes
- If you change the embedding model, re‑run the prefetch for the new model name.

Ports
- Frontend: 80
- API: 8000 (proxied through Nginx at /api)
- Kiwix: 8080 (proxied through Nginx at /kiwix)
- Ollama (host): 11434 (OpenAI-compatible API)

One-liner bootstrap (everything except downloading the ZIM)
- `sh -c 'mkdir -p data/zims && docker compose up -d nginx kiwix api; if [ ! -f data/zims/enwiki.zim ]; then echo "Download the ZIM to ./data/zims/enwiki.zim or run: make prefetch-zim"; fi'`

## Makefile shortcuts

Core tasks are available via `make`:

```
# Start/Stop & logs
make up                   # start nginx+kiwix+api
make down                 # stop all services
make recreate-api         # recreate API container (pick up code without rebuild)
make logs-api             # tail API logs
make api-health           # GET /api/health

# ZIM + title index
make prefetch-zim               # downloads Wikipedia ZIM into ./data/zims/enwiki.zim (tries two mirrors)
make build-title-index          # builds titles.sqlite inside API (use LIMIT=... for quick partial)

# Frontend dev
make serve-frontend             # dev-serve ./frontend at http://localhost:5173

# Data bundles
make package-data               # bundle ZIM + title index into dist/wiki-in-a-box-data.tar.gz
make verify-data                # verify data bundle hashes
```


Run Ollama on the host and the API will reach it at `http://host.docker.internal:11434/v1` (macOS for the demo)
on Linux, ensure `OLLAMA_HOST=0.0.0.0` and that port 11434 is open in any firewall.

Notes
- Linux support is baked in: this repo sets `extra_hosts: ["host.docker.internal:host-gateway"]` under the `api` service.
- The API expects `LLM_MODEL=gpt-oss:20b` by default

## API (minimal)

- `GET /api/health` → `{ wan, model_name, indices }`
- `POST /api/chat` body `{ question, mode?: 'default'|'eli10'|'advanced'|'quiz', k?: number }` — streams SSE

Examples
- Health: `curl http://localhost/api/health`
- Chat (SSE): `curl -N -X POST http://localhost/api/chat -H 'content-type: application/json' -d '{"question":"Why is the sky blue?"}'`

## What is a ZIM? How to get it

- ZIM is an offline content format used by Kiwix. Wikipedia “ZIM” files bundle articles for offline use.
- For speed/size, download an English “no-images” Wikipedia ZIM (e.g., enwiki_en_all_nopic.zim) from Kiwix downloads. Put it at `./data/zims/enwiki.zim`.

Expected paths
- ZIM: `./data/zims/enwiki.zim` (host) → `/data/zims/enwiki.zim` (container)
  - To use a different filename, either rename your file to `enwiki.zim` or set `ZIM_PATH` in `docker-compose.yml` to match your chosen path.

## Hardware notes (pragmatic)

- Recommended: NVIDIA GPU with ≥16 GB VRAM (e.g., A10G 24 GB, RTX 4090) for smooth gpt‑oss:20b streaming. CPU-only works but is slower.
- CPU/RAM: 8+ cores, 16–32 GB RAM, SSD strongly recommended (ZIM reads and per‑query chunking/embedding).
- Disk: ZIM (no-images) tens of GB.

## How it works (Hybrid no‑index RAG)

- Recall: libzim full‑text search over the local ZIM file to shortlist relevant article paths.
- Chunk: parse HTML → text, then create overlapping chunks (defaults below).
- Embed + rank: embed chunks and the query via SentenceTransformers (BAAI/bge-small-en-v1.5 by default), score via cosine similarity, take top‑K.
- Cite: return [1][2] style citations that link to `/kiwix/<path>` (served by kiwix-serve).

Title‑first recall (parent‑document)
- We leverage Wikipedia’s strong titles. For a query, we fetch title suggestions (libzim SuggestionSearcher), then build context from those pages first (section‑aware chunking), and semantically rank chunks. If the match is strong enough, we answer directly (early exit).
  - Optional: build a lightweight SQLite FTS5 title index for better title recall (see below).
  - Page‑level re‑ranker (title + lead): before chunking, we embed the page's title and lead paragraph for each candidate title and semantically re‑rank pages; we then deep‑dive only into the top `TOP_PAGES` pages. This reduces noise and focuses chunking on the most relevant topics.

Full‑text fallback + second‑pass (auto)
- If the semantic match looks weak (low top similarity), we automatically widen recall (higher candidate caps) and retry the rank.
- This improves coverage without a full prebuilt dense index.

Tuning (env vars)
- `EMBED_MODEL` (default `BAAI/bge-small-en-v1.5`)
- `MAX_ARTICLES` (default `20`) — number of recalled articles per query
- `CHUNK_TOKENS` (default `160`) and `CHUNK_OVERLAP` (default `20`)
- `MAX_CHUNKS` (default `120`) — global chunk cap per query
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — enforce offline model loading
  - `DEBUG_REASONING` (default `true`) — when `true`, stream reasoning frames as `event: reasoning`
  - `SECOND_PASS_ENABLE` (default `true`) — widen recall if evidence is weak
  - `SIM_THRESHOLD` (default `0.22`) — similarity threshold for second pass
  - `SECOND_PASS_FACTOR` (default `2.0`) — how much to widen recall caps
  - `EMBED_LRU_SIZE` (default `64`) — in‑memory cache of per‑article chunk embeddings
  - `SUGGESTION_LIMIT` (default `20`) — cap number of title suggestions to consider
  - `TITLE_SIM_EXIT` (default `0.28`) — early‑exit threshold when title path yields strong match
  - `TITLE_INDEX_DIR` (default `/data/title_index`) — where the SQLite title index lives
  - `TOP_PAGES` (default `3`) — number of top pages (by title+lead similarity) to deep‑dive into

Streaming & Reasoning
- Answers stream as SSE `event: token` frames; end with `event: done`.
- When `DEBUG_REASONING=true`, if the model returns `delta.reasoning` frames (gpt‑oss supports this), they emit as `event: reasoning`.

Answering when context is missing
- If the provided CONTEXT doesn’t contain the answer, the assistant provides a concise best‑effort explanation prefixed with `General (no local cite):` instead of declining.

Ask about a specific page
- `POST /api/answer_from_page` with body `{ "path": "/Sunset", "question": "...", "k": 6 }` — builds CONTEXT from that page only and answers.

## Notes

For stronger title recall beyond prefix suggestions, build a tiny SQLite FTS index of titles:

```
make build-title-index
```

This scans the ZIM once and writes `titles.sqlite` under `TITLE_INDEX_DIR` (default `/data/title_index`).
Subsequent queries will combine SuggestionSearcher with FTS5 title search for robust title recall.

What “Build FTS first (fast)” means
- We only index article titles and paths into a single small SQLite file using FTS5 (BM25 ranking). No embeddings and no article content are stored.
- Building this title index is dramatically faster and lighter than a full dense index; it’s a one‑pass scan over ZIM entries that writes (title, path) pairs.
- You can control scope with `LIMIT`, e.g. index the first 200k entries quickly for a demo, then rebuild without `LIMIT` for full coverage.

Pipeline order (at query time)
1. Title FTS + libzim suggestions → fetch candidate pages
2. Page‑level re‑rank (embed: title + lead paragraph) → keep `TOP_PAGES`
3. Section‑aware chunking on those pages → embed sections → semantic re‑rank
4. Early‑exit if the best match is strong (`TITLE_SIM_EXIT`)
5. Otherwise, fallback to full‑text recall; if evidence is weak, widen with second‑pass

Why titles first
- Wikipedia excels at concept titles (e.g., “Sunset”, “Rayleigh scattering”). Parent‑document retrieval over those titles yields the right context quickly.
- Section‑aware chunks give focused snippets and better re‑ranking for grounded answers.
