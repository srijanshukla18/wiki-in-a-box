# Wiki‑in‑a‑Box — Follow‑ups and Future Work

This document captures concrete, non‑hacky improvements we deferred for the submission. Items are ordered by impact vs. effort.

## 1) Robust Title Recall Without Hardcoded Phrases

Goal: Avoid conversational noise (e.g., "tell me about …") and steer title recall toward the intended topic without hand‑tuned rules.

Plan (corpus‑driven):
- Tokenize query generically (letters/numbers, lowercase). Do not strip phrases.
- Compute title document frequency df(t) on the fly via the SQLite FTS5 title index:
  - SELECT COUNT(*) FROM titles WHERE titles MATCH ?  (for token t)
- Select discriminative tokens/variants by IDF (e.g., top 2–4 tokens by 1/df, or df/total_docs < τ).
- Generate algorithmic variants (no hardcode):
  - Concatenate adjacent short tokens (≤3 chars) to capture cases like "open ai" → "openai"; consider hyphenated form too.
  - Evaluate variants by df; keep those that are rare/high‑IDF.
- Build an FTS query as an OR‑joined expression of selected tokens/variants (e.g., "openai" OR open OR ai).
- Log for transparency: [title-query] tokens=... variants=... fts='...'.

Title closeness guard (no early exit on wrong pages):
- Compute a lightweight title closeness score for each candidate (e.g., token Jaccard or SequenceMatcher on normalized title/path vs. query tokens).
- Combine with the current page‑level semantic reranker (title+lead cosine).
- Only early‑exit if BOTH:
  - Section‑level similarity ≥ TITLE_SIM_EXIT
  - Title closeness ≥ MIN_TITLE_CLOSENESS

Env knobs to add:
- MAX_TITLE_TOKENS (e.g., 4) — number of highest‑IDF tokens to keep
- MIN_TITLE_CLOSENESS (e.g., 0.2–0.3) — guard threshold for early exit

Acceptance:
- Query "tell me about open ai" routes to OpenAI consistently, with correct citations.
- Zero hardcoded phrase lists.

## 2) Optional Incremental FAISS (Titles Only)

Goal: Semantic title recall without pre‑embedding the entire corpus.

Plan:
- Keep FTS as the primary title recall.
- For the top‑N FTS candidates per query, embed their title+lead on‑demand; insert vectors into a small, persisted FAISS index keyed by path.
- Over time, frequently‑hit titles are cached semantically; future queries get fast semantic reranking even before page‑level lead extraction.

Env knobs:
- FAISS_TITLES_PATH (opt‑in)
- FAISS_TITLES_CAP (max vectors)

## 3) UI Signals for Retrieval Phases (Extended)

Current: we show "Searching titles …" → "Found N sources. Generating with …".

Add (optional):
- Stream small info events to surface fallback/widening:
  - [fallback] title best=… < threshold → show "Broadened search (full text) …"
  - [second-pass] widening recall (best=…) → show "Widening search …"
- Keep off by default; enable with DEBUG_UI_FLOW=true.

## 4) Makefile: unpack & smoke test

Add:
- make unpack-data — untar dist/wiki-in-a-box-data.tar.gz into ./
- make smoke-test — curl /api/health and a known query; exit non‑zero on failure

## 5) API: Debug endpoints (opt‑in)

- GET /api/debug/title_query?q=... → returns tokens, variants, FTS string, top candidates (no SSE)
- GET /api/debug/citations?q=... → non‑streaming citation preview for QA
- Guard behind DEBUG_ROUTES=true

## 6) Section Chunking Refinements

- Prefer lead + top N sections by tf–idf against query tokens before embedding all sections (saves latency on long pages).
- Headings normalisation; skip tables and ref‑heavy blocks.

## 7) Packaging Quality of Life

- Add make unpack-data (see #4).
- Add make doctor to check:
  - ZIM presence/size
  - titles.sqlite presence
  - Ollama reachability from inside the container
  - API health

## 8) Structured Outputs (Quiz Mode)

- Switch quiz to enforce strict JSON schema with local validation/repair.
- Optionally wire model structured outputs if/when available in the backend.

## 9) Agent‑Ready Tooling (Future)

- Expose a fetch_article(path) tool so the model can request specific pages mid‑generation.
- Useful for multi‑hop or when the first context is insufficient.

## Notes

- All of the above avoid hardcoded phrase lists; they are corpus‑driven, explainable, and configurable.
- Title‑first → page‑lead semantic rerank → section‑aware chunks remains the backbone; these items make it more robust and general.
