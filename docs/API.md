# Wiki‑in‑a‑Box API Specification

This document describes the public HTTP API exposed by the Wiki‑in‑a‑Box backend (formerly Wiki‑Lantern).

- Base URL (direct): `http://localhost:8000`
- Base URL (via Nginx): `http://localhost` with API under `/api`
- Authentication: none (intended for local/LAN use)
- Content types: JSON (requests), Server‑Sent Events (responses for chat endpoints)

## Conventions

- All request bodies are JSON with `application/json` content type.
- Streaming responses use Server‑Sent Events (SSE) with `text/event-stream`.
- Citations link to Kiwix through the Nginx proxy under `/kiwix/...`.

---

## GET /api/health

Returns basic service status and index presence.

- Request: none
- Response 200 (application/json):
```json
{ "wan": true, "model_name": "gpt-oss:20b", "indices": { "zim_present": true } }
```
- Errors: 500 on unexpected server errors

---

## POST /api/chat (SSE)

Answers a question grounded in locally stored Wikipedia content. Returns a streaming SSE response.

- Request (application/json):
```json
{ "question": "Why is the sky orange at sunset?", "k": 6, "mode": "default" }
```

- Response (text/event-stream): a sequence of SSE frames. Event types:
  - `event: citations` (emitted first)
  ```json
  { "citations": [ {"id": 1, "title": "Sunset", "url": "/kiwix/Sunset", "snippet": "..."} ] }
  ```
  - `event: reasoning` (optional; only if DEBUG_REASONING=true)
  ```json
  { "token": "partial reasoning text ..." }
  ```
  - `event: token` (answer tokens)
  ```json
  { "token": "partial answer text ..." }
  ```
  - `event: done`
  ```json
  {}
  ```

- Status codes: 200 (streaming), 400 (bad request), 502 (LLM error), 503 (index not loaded)
- Notes: title‑first retrieval → section‑aware chunking → semantic re‑rank → early exit; fallback to full‑text and second‑pass if weak. If context is insufficient, reply with `General (no local cite): ...`.

Example:
```bash
curl -N -X POST http://localhost:8000/api/chat   -H 'content-type: application/json'   -d '{"question":"Why is the sky orange at sunset?"}'
```

---

## POST /api/answer_from_page (SSE)

Answers a question using content from a single Wikipedia page (parent‑document retrieval).

- Request (application/json):
```json
{ "path": "/Sunset", "question": "Why is the sky orange?", "k": 6, "mode": "default" }
```
- Response: SSE frames identical to `/api/chat`.
- Status codes: same as `/api/chat`, plus 400 if `path` missing/invalid.

Example:
```bash
curl -N -X POST http://localhost:8000/api/answer_from_page   -H 'content-type: application/json'   -d '{"path":"/Sunset", "question":"Why is the sky orange?", "k":6}'
```

---

## Server configuration & tuning

(see docker‑compose.yml for defaults)

- Retrieval & chunking: `MAX_ARTICLES`, `MAX_CHUNKS`, `CHUNK_TOKENS`, `CHUNK_OVERLAP`, `RECALL_LIMIT`
- Title‑first recall: `SUGGESTION_LIMIT`, `TITLE_SIM_EXIT`, `TITLE_INDEX_DIR`
- Second‑pass widening: `SECOND_PASS_ENABLE`, `SIM_THRESHOLD`, `SECOND_PASS_FACTOR`
- Caches: `EMBED_LRU_SIZE`, `SUGGEST_LRU_SIZE`
- LLM & streaming: `LLM_BASE`, `LLM_MODEL`, `MAX_CONTEXT_TOKENS`, `MAX_NEW_TOKENS`, `DEBUG_REASONING`
- Paths: `ZIM_PATH`

---

## Building the title index (SQLite FTS5)

A small, fast index of Wikipedia titles improves title‑first recall.

- Build:
```bash
make build-title-index                 # full scan
LIMIT=200000 make build-title-index    # partial for quick demo
```
- Writes `titles.sqlite` to `TITLE_INDEX_DIR` (default `/data/title_index`). Retrieval uses it automatically if present.

---

## Error handling notes

- If the upstream LLM yields no streamed tokens, the server performs a one‑shot non‑streaming fallback once and emits the content if available.
- If the model returns an error/invalid payload, the server logs a brief diagnostic and closes the stream.

---

## Compatibility

- Kiwix must be running and the ZIM mounted at `ZIM_PATH`.
- Ollama must be running on the host with `LLM_MODEL` available; the API reaches it via `host.docker.internal:11434` (Linux users should set `OLLAMA_HOST=0.0.0.0`).
