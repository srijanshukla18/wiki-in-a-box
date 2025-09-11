# Wiki‑in‑a‑Box: High‑Quality, Fully‑Offline Wikipedia RAG Without Pre‑Indexing

## Abstract
- Wikipedia is a uniquely hard corpus for RAG: it’s huge, inconsistently structured, and title‑rich. “Chunk‑everything + dense index” works but is heavy (hours to build, tens of GB) and brittle on low‑end hardware.
- Wiki‑in‑a‑Box delivers fast, grounded answers with citations on commodity machines, fully offline, without pre‑embedding the corpus. The core idea: use Wikipedia’s strongest signal (its titles) first, then deepen only where needed.

## Motivation
- Internet‑in‑a‑box: reliable answers and citations when offline, on modest hardware, with a single ZIM file.
- Near‑zero setup time: start serving immediately; optional lightweight indexing that completes in minutes, not hours.
- Respect the corpus: parent‑document retrieval over curated topic pages beats blind chunking.

## Challenges with Wikipedia as a KB
- Inconsistent information architecture, variable writing discipline, duplication across articles.
- Lexical search can miss obvious intent (“orange sky at sunset” → “Sunset”, “Rayleigh scattering”).
- Full dense pre‑indexing is expensive (compute, time, disk) and hard to maintain offline.

## Design Goals
- Quality: retrieve the right pages, then the right sections, and ground the answer.
- Speed: sub‑second to a few seconds on an Apple M‑class laptop or similar CPU.
- Offline: all retrieval and inference local; optional small, fast index only.
- Transparency: stream citations first; never fabricate references.

## Approach (Query‑Time, No Pre‑Embedding)

### Title‑First Retrieval (Primary)
- Titles are Wikipedia’s strongest “topic” signal. We combine:
  - A lightweight SQLite FTS5 index of (title, path) (built in minutes).
  - libzim’s SuggestionSearcher as a complementary title prefix source.
- Results are deduplicated and treated as candidate pages.

### Page‑Level Semantic Re‑Ranking (Title + Lead)
- For each candidate page, we embed “title + lead paragraph” once, then compute cosine similarity with the query embedding.
- Keep only the top P pages (default 3). This filters out near‑name collisions (“Sky …”) and focuses RAG on truly relevant topics.
- Small in‑memory LRU cache stores these lead embeddings.

### Section‑Aware Chunking (Parent‑Document Deep‑Dive)
- Parse HTML headings (h2/h3) to form meaningful sections with a “lead” section.
- Long sections are windowed; short sections are used as‑is.
- Each section is embedded; we re‑rank sections to assemble the final context.

### Early Exit and Progressive Fallback
- If top section similarity is strong (configurable threshold), answer immediately with those citations.
- Otherwise, fallback to a hybrid full‑text recall → chunk+embed → re‑rank.
- If evidence remains weak, automatically widen recall (second pass) and try again.

### Honest Output and UX
- Citations stream first (SSE) so the client can render chips immediately.
- If the context is insufficient to answer, return “General (no local cite): …” — no fabricated citations.
- Optional endpoint “answer_from_page” answers using a single page (deterministic parent‑document retrieval).

## Architecture Overview
- ZIM access via python‑libzim: Archive, Searcher, SuggestionSearcher.
- Title index: SQLite FTS5 (titles.sqlite). Titles only — tiny and fast to build.
- Dense embeddings: SentenceTransformers (BAAI/bge‑small‑en‑v1.5) for query, page‑lead, and section chunks only (on‑demand).
- LLM: gpt‑oss via Ollama (OpenAI Chat Completions‑compatible), “reasoning_effort = low”, streamed answers; optional reasoning frames.
- API: FastAPI SSE endpoints; Nginx proxy; Kiwix for in‑browser article views.

## Implementation Details
### Title Index (FTS5)
- Build once: `make build-title-index` ( `LIMIT=…` for fast partial).
- Query with an OR‑joined token query (“sunset OR sky OR orange”), not raw prose, for robust FTS matches.

### Page‑Level Reranker
- Embed “title + lead paragraph” for each title candidate; cache lead embeddings in memory keyed by path.
- Keep `TOP_PAGES` only (default 3) for section‑level deep‑dive.

### Section Chunking
- Strip boilerplate (infobox, navboxes) and chunk by h2/h3; lead section is text before the first heading.
- Short sections stay intact; long sections are windowed (160 tokens, 20 overlap by default).

### Early Exit Thresholds
- `TITLE_SIM_EXIT` controls early exit quality for title‑first path (default 0.28).
- `SIM_THRESHOLD` triggers second‑pass widening (default 0.22).

### Progressive Recall
- If the title‑first path isn’t strong enough, fall back to full‑text recall with widening (factor 2.0 by default).

## Offline Operation
- All components run locally. No WAN required after one‑time model prefetch.
- Hugging Face model cache mounted read‑only; HF offline flags set.
- Title index is a tiny SQLite file; optional, but recommended for highest quality.

## Performance & Latency (Expected)
- Title index build: minutes on laptop (titles only; small file). Partial builds with `LIMIT` for quick demos.
- Query latency:
  - Title+lead rerank + section chunking for 2–3 pages: typically 0.5–2.0s on CPU.
  - Full‑text fallback and second‑pass adds ~0.5–1.0s when needed.
- Memory: small in‑memory caches speed repeated queries; no large on‑disk vector store.

## Quality Behavior (What You See)
- “Why is the sky orange at sunset?”:
  - Title FTS: Sunset, Rayleigh scattering, Atmospheric optics … → Page‑lead rerank selects “Sunset” and friends.
  - Section chunks: “Causes of the red/orange sky at sunset” → strong early exit with correct citations.
- “When was Einstein born?”:
  - Title FTS: Albert Einstein → Page‑lead rerank keeps Einstein → Lead/biography section chunk → early exit.

## Why This Beats “Chunk Everything”
- Doesn’t rely on embedding millions of chunks upfront. Instead, it uses Wikipedia’s curation — titles and lead paragraphs — as a principled index layer.
- Parent‑document retrieval respects “topics” like human‑curated docs do; section‑aware chunks keep context focused and tight.
- The dense embedding work is only where needed: the query, the title+lead of a handful of pages, and sections from those pages.

## Limitations & Tradeoffs
- Title‑first is not perfect: some queries still need full‑text fallback (we do this automatically).
- Section parsing is heuristic; embedded HTML edge cases can require tuning.
- No persistent semantic title index by default; we cache leads in memory only (keeps footprint small).

## Future Work
- Incremental semantic title FAISS (optional): embed top N titles over time or on demand; persist for faster, stronger title reranking.
- Category/taxonomy bias: use Wikipedia categories/infobox links to boost likely title candidates.
- Better ranking signals: add a lightweight BM25/semantic fusion or learned scoring atop section chunks.
- Tool use: expose a “fetch_article(path)” tool to let the model request deeper context mid‑generation when warranted.

## Operator UX & Dev Notes
- Makefile shortcuts for up/down/logs, title index build, API health, Ollama checks.
- Docker compose mounts code; force‑recreate (not rebuild) to pick up changes.
- API spec in `docs/API.md`; README gives an end‑to‑end offline story.

## Conclusion
- Wiki‑in‑a‑Box shows that Wikipedia RAG doesn’t need to start with “embed the universe.” Title‑first recall, a tiny title index, page‑level reranking, and section‑aware chunking solve most of the hard parts — fast, offline, and on modest machines.
- The result is a clean, principled retrieval pipeline that aligns with how humans find and consume Wikipedia: start at the right page, then drill down to the right section, and only embed what you need.
