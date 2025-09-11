import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
from libzim.reader import Archive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher
from title_index import search_titles


EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "20"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "160"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "120"))
# Upper bound on recalled paths (single query) for full-text fallback
RECALL_LIMIT = int(os.getenv("RECALL_LIMIT", "200"))
# Second-pass recall when evidence looks weak
SECOND_PASS_ENABLE = os.getenv("SECOND_PASS_ENABLE", "true").lower() == "true"
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.22"))
SECOND_PASS_FACTOR = float(os.getenv("SECOND_PASS_FACTOR", "2.0"))
# Title suggestions and early-exit threshold
SUGGESTION_LIMIT = int(os.getenv("SUGGESTION_LIMIT", "20"))
TITLE_SIM_EXIT = float(os.getenv("TITLE_SIM_EXIT", "0.28"))
# Simple in-memory LRU cache sizes
EMBED_LRU_SIZE = int(os.getenv("EMBED_LRU_SIZE", "64"))
SUGGEST_LRU_SIZE = int(os.getenv("SUGGEST_LRU_SIZE", "128"))
TOP_PAGES = int(os.getenv("TOP_PAGES", "3"))
TITLE_INDEX_DIR = os.getenv("TITLE_INDEX_DIR", "/data/title_index")


@dataclass
class RetrievalItem:
    id: int
    title: str
    url: str
    snippet: str
    score: float


def _normalize_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["nav", "footer", "header"]:
        for tag in soup.select(sel):
            tag.decompose()
    text = soup.get_text(" ")
    return " ".join(text.split())


def _chunks(words: List[str], size_tokens: int, overlap: int) -> Iterable[Tuple[int, int]]:
    n = len(words)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(n, start + size_tokens)
        yield start, end
        if end == n:
            break
        start = max(0, end - overlap)


_qe_cache: dict = {}


class _LRU:
    def __init__(self, cap: int):
        self.cap = max(0, cap)
        self._d: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> Optional[dict]:
        if self.cap == 0:
            return None
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def set(self, key: str, val: dict) -> None:
        if self.cap == 0:
            return
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = val
        else:
            self._d[key] = val
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

def _normalize_query(q: str) -> str:
    return " ".join(q.split()).strip().lower()


class Retriever:
    def __init__(self, zim_path: str, kiwix_base: str):
        self.zim_path = zim_path
        self.kiwix_base = kiwix_base.rstrip("/")
        if not os.path.exists(self.zim_path):
            raise FileNotFoundError(f"ZIM not found at {self.zim_path}")
        self.zim = Archive(self.zim_path)
        # Load encoder once; relies on offline cache if configured
        self.model = SentenceTransformer(EMBED_MODEL)
        self._chunk_cache = _LRU(EMBED_LRU_SIZE)
        self._suggest_cache = _LRU(SUGGEST_LRU_SIZE)
        self._lead_cache = _LRU(EMBED_LRU_SIZE)

    def _extract_chunks(self, entry) -> tuple[list[str], list[tuple[str, str, str]]]:
        item = entry.get_item()
        mt = (item.mimetype or "").lower()
        if not (mt.startswith("text/") or "html" in mt):
            return [], []
        html = bytes(item.content)
        soup = BeautifulSoup(html, "html.parser")
        # Remove boilerplate tags
        for sel in ["nav", "footer", "header", ".infobox", ".navbox", ".metadata"]:
            for tag in soup.select(sel):
                tag.decompose()
        # Build sections by h2/h3. Lead section is text before first h2/h3.
        sections: list[tuple[str, str]] = []  # (heading, text)
        def clean(txt: str) -> str:
            return " ".join(txt.split())
        # collect in document order
        current_title = "Lead"
        current_parts: list[str] = []
        def flush():
            nonlocal current_title, current_parts
            if current_parts:
                sections.append((current_title, clean(" ".join(current_parts))))
                current_parts = []
        for el in soup.body.find_all(recursive=False) if soup.body else soup.find_all(recursive=False):
            # search through descendants to catch headings and paragraphs in flow
            for node in el.descendants:
                if getattr(node, "name", None) in ("h2", "h3"):
                    flush()
                    current_title = clean(node.get_text(" ")) or current_title
                elif getattr(node, "name", None) in ("p", "li"):
                    txt = node.get_text(" ")
                    if txt:
                        current_parts.append(txt)
        flush()
        title = entry.title or entry.path
        url = "/" + entry.path.lstrip("/")
        chunk_texts: list[str] = []
        chunk_meta: list[tuple[str, str, str]] = []
        # Turn sections into chunks; for long sections, fall back to token slicing
        for sec_title, sec_text in sections:
            if not sec_text:
                continue
            words = sec_text.split()
            if len(words) > CHUNK_TOKENS * 2:
                for s, t in _chunks(words, CHUNK_TOKENS, CHUNK_OVERLAP):
                    chunk = " ".join(words[s:t])
                    snippet = " ".join(words[s : min(t, s + 60)])
                    chunk_texts.append(chunk)
                    chunk_meta.append((f"{title} — {sec_title}", url, snippet))
                    if len(chunk_texts) >= MAX_CHUNKS:
                        break
            else:
                snippet = " ".join(words[:60])
                chunk_texts.append(" ".join(words))
                chunk_meta.append((f"{title} — {sec_title}", url, snippet))
            if len(chunk_texts) >= MAX_CHUNKS:
                break
        return chunk_texts, chunk_meta

    def _extract_lead(self, entry) -> tuple[str, str]:
        item = entry.get_item()
        mt = (item.mimetype or "").lower()
        if not (mt.startswith("text/") or "html" in mt):
            return entry.title or entry.path, ""
        html = bytes(item.content)
        soup = BeautifulSoup(html, "html.parser")
        for sel in ["nav", "footer", "header", ".infobox", ".navbox", ".metadata"]:
            for tag in soup.select(sel):
                tag.decompose()
        lead_parts: list[str] = []
        def clean(txt: str) -> str:
            return " ".join(txt.split())
        for node in soup.body.descendants if soup.body else soup.descendants:
            name = getattr(node, "name", None)
            if name in ("h2", "h3"):
                break
            if name in ("p", "li"):
                txt = node.get_text(" ")
                if txt:
                    lead_parts.append(txt)
        lead = clean(" ".join(lead_parts))
        return (entry.title or entry.path), lead

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")

    def _recall_and_chunks(self, queries: list[str], wanted: int, recall_limit: int) -> tuple[list[str], list[tuple[str,str,str]], np.ndarray]:
        paths: list[str] = []
        seen_paths = set()
        for qstr in queries:
            try:
                res = Searcher(self.zim).search(Query().set_query(qstr))
                for p in list(res.getResults(0, wanted)):
                    if p not in seen_paths:
                        paths.append(p)
                        seen_paths.add(p)
                        if len(paths) >= recall_limit:
                            break
                if len(paths) >= recall_limit:
                    break
            except Exception:
                continue

        all_texts: list[str] = []
        all_meta: list[tuple[str,str,str]] = []
        all_embs: list[np.ndarray] = []

        for p in paths:
            if len(all_texts) >= MAX_CHUNKS:
                break
            cached = self._chunk_cache.get(p)
            if cached is not None:
                texts = cached.get("texts", [])
                meta = cached.get("meta", [])
                emb = cached.get("emb", None)
            else:
                try:
                    entry = self.zim.get_entry_by_path(p)
                except Exception:
                    continue
                texts, meta = self._extract_chunks(entry)
                emb = None
            # encode if needed and if we still have budget
            if texts:
                take = min(len(texts), MAX_CHUNKS - len(all_texts))
                use_texts = texts[:take]
                use_meta = meta[:take]
                if emb is None:
                    emb = self._encode(texts)
                use_emb = emb[:take]
                all_texts.extend(use_texts)
                all_meta.extend(use_meta)
                all_embs.append(use_emb)
                # store in cache
                if cached is None:
                    self._chunk_cache.set(p, {"texts": texts, "meta": meta, "emb": emb})

        XV = np.vstack(all_embs) if all_embs else np.zeros((0, 384), dtype="float32")
        return all_texts, all_meta, XV

    def _title_suggest_paths(self, q: str) -> list[str]:
        qn = _normalize_query(q)
        cached = self._suggest_cache.get(qn)
        if cached is not None:
            return cached.get("paths", [])
        # Build candidate prefixes from the query (tokens and small n-grams)
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", qn)
        stop = {
            "the","a","an","and","or","of","to","for","with","in","on","by","at","is","are","was","were",
            "be","been","being","what","why","how","who","when","where","which","that","this","these","those"
        }
        tokens = [w for w in words if w not in stop and len(w) >= 3]
        # Compose prefixes: single tokens + bigrams + last word (often key)
        cands: list[str] = []
        seen_c = set()
        for w in tokens:
            if w not in seen_c:
                cands.append(w)
                seen_c.add(w)
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            if bg not in seen_c:
                cands.append(bg)
                seen_c.add(bg)
        if tokens:
            last = tokens[-1]
            if last not in seen_c:
                cands.append(last)
                seen_c.add(last)
        # Query SuggestionSearcher with candidates until we fill up
        out: list[str] = []
        try:
            ss = SuggestionSearcher(self.zim)
            for cand in cands:
                try:
                    sug = ss.suggest(cand)
                    count = getattr(sug, "getEstimatedMatches", lambda: SUGGESTION_LIMIT)()
                    take = min(SUGGESTION_LIMIT, int(count) if isinstance(count, int) else SUGGESTION_LIMIT)
                    res = sug.getResults(0, take)
                    for p in res:
                        out.append(p)
                        if len(out) >= SUGGESTION_LIMIT:
                            break
                    if len(out) >= SUGGESTION_LIMIT:
                        break
                except Exception:
                    continue
        except Exception:
            pass
        # Also query FTS5 title index if present, using an OR-joined token query
        try:
            token_query = " OR ".join(tokens) if tokens else qn
            fts_rows = search_titles(TITLE_INDEX_DIR, token_query, limit=max(10, SUGGESTION_LIMIT))
            for path, _title in fts_rows:
                out.append(path)
        except Exception:
            pass
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for p in out:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        self._suggest_cache.set(qn, {"paths": deduped})
        try:
            print(f"[title] suggestions for '{qn}': {deduped[:SUGGESTION_LIMIT]}", flush=True)
        except Exception:
            pass
        return deduped

    def search(self, query: str, top_k: int = 6) -> List[RetrievalItem]:
        # 1) Title-based parent-document retrieval
        title_paths = self._title_suggest_paths(query)
        if title_paths:
            # 1.5) Page-level semantic re-ranker on title candidates (title + lead paragraph)
            qv = self._encode([query])
            page_scores: list[tuple[float, str, str]] = []  # (score, path, title)
            for p in title_paths:
                cached = self._lead_cache.get(p)
                if cached is not None:
                    lead_emb = cached.get("emb")
                    title_str = cached.get("title", p)
                else:
                    try:
                        entry = self.zim.get_entry_by_path(p)
                    except Exception:
                        continue
                    title_str, lead = self._extract_lead(entry)
                    if not lead:
                        continue
                    # concatenate title + lead for a strong page signal
                    page_text = f"{title_str}. {lead}"
                    lead_emb = self._encode([page_text])  # (1,d)
                    self._lead_cache.set(p, {"emb": lead_emb, "title": title_str})
                if lead_emb is None:
                    continue
                s = float((lead_emb @ qv.T).ravel()[0])
                page_scores.append((s, p, title_str))
            if page_scores:
                page_scores.sort(key=lambda x: -x[0])
                top_pages = page_scores[: max(1, TOP_PAGES)]
                try:
                    preview = [(t, round(s, 3)) for s, _p, t in top_pages]
                    print(f"[page-rerank] top pages: {preview}", flush=True)
                except Exception:
                    pass
                # Build section chunks only for selected pages
                all_texts: list[str] = []
                all_meta: list[tuple[str,str,str]] = []
                all_embs: list[np.ndarray] = []
                for _s, p, _t in top_pages:
                    if len(all_texts) >= MAX_CHUNKS:
                        break
                    cached = self._chunk_cache.get(p)
                    if cached is not None:
                        texts = cached.get("texts", [])
                        meta = cached.get("meta", [])
                        emb = cached.get("emb", None)
                    else:
                        try:
                            entry = self.zim.get_entry_by_path(p)
                        except Exception:
                            continue
                        texts, meta = self._extract_chunks(entry)
                        emb = None
                    if not texts:
                        continue
                    take = min(len(texts), MAX_CHUNKS - len(all_texts))
                    use_texts = texts[:take]
                    use_meta = meta[:take]
                    if emb is None:
                        emb = self._encode(texts)
                    use_emb = emb[:take]
                    all_texts.extend(use_texts)
                    all_meta.extend(use_meta)
                    all_embs.append(use_emb)
                    if cached is None:
                        self._chunk_cache.set(p, {"texts": texts, "meta": meta, "emb": emb})
                if all_texts:
                    XV = np.vstack(all_embs) if all_embs else np.zeros((0, 384), dtype="float32")
                    scores = (XV @ qv.T).ravel()
                    best = float(scores.max()) if scores.size else 0.0
                    if best >= TITLE_SIM_EXIT:
                        top = np.argsort(-scores)[: max(top_k, 1)]
                        items: List[RetrievalItem] = []
                        for i, idx in enumerate(top.tolist()):
                            title, url, snippet = all_meta[idx]
                            items.append(RetrievalItem(id=i, title=title, url=url, snippet=snippet, score=float(scores[idx])))
                        try:
                            print(f"[early-exit] title path best={best:.3f}", flush=True)
                        except Exception:
                            pass
                        return items[:top_k]
                    else:
                        try:
                            print(f"[fallback] title best={best:.3f} < {TITLE_SIM_EXIT:.2f}, using full-text", flush=True)
                        except Exception:
                            pass

        # 2) Full-text hybrid recall (existing)
        queries = [query]
        wanted = max(200, MAX_ARTICLES * 10)
        chunk_texts, chunk_meta, XV = self._recall_and_chunks(queries, wanted=wanted, recall_limit=RECALL_LIMIT)
        if len(chunk_texts) == 0:
            return []

        try:
            qv = self._encode([query])  # (1,d)
            scores = (XV @ qv.T).ravel()
            best = float(scores.max()) if scores.size else 0.0
            # Second pass if evidence looks weak
            if SECOND_PASS_ENABLE and best < SIM_THRESHOLD:
                print(f"[second-pass] widening recall (best={best:.3f})", flush=True)
                widened_wanted = int(wanted * SECOND_PASS_FACTOR)
                widened_limit = int(RECALL_LIMIT * SECOND_PASS_FACTOR)
                chunk_texts, chunk_meta, XV = self._recall_and_chunks(queries, wanted=widened_wanted, recall_limit=widened_limit)
                if len(chunk_texts) == 0:
                    return []
                qv = self._encode([query])
                scores = (XV @ qv.T).ravel()
            top = np.argsort(-scores)[: max(top_k, 1)]
        except Exception:
            # Fallback: no embeddings available; return first textual chunks
            top = np.arange(min(len(chunk_texts), top_k))
            scores = np.ones(len(chunk_texts), dtype="float32")

        items: List[RetrievalItem] = []
        for i, idx in enumerate(top.tolist()):
            title, url, snippet = chunk_meta[idx]
            items.append(
                RetrievalItem(
                    id=i,
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=float(scores[idx]),
                )
            )
        return items[:top_k]

    def search_in_path(self, path: str, query: str, top_k: int = 6) -> List[RetrievalItem]:
        # Build chunks + embeddings for a single page and rank
        try:
            entry = self.zim.get_entry_by_path(path.lstrip("/"))
        except Exception:
            return []
        cached = self._chunk_cache.get(entry.path)
        if cached is not None:
            texts = cached.get("texts", [])
            meta = cached.get("meta", [])
            emb = cached.get("emb", None)
        else:
            texts, meta = self._extract_chunks(entry)
            emb = None
        if not texts:
            return []
        if emb is None:
            emb = self._encode(texts)
            self._chunk_cache.set(entry.path, {"texts": texts, "meta": meta, "emb": emb})
        qv = self._encode([query])
        scores = (emb @ qv.T).ravel()
        top = np.argsort(-scores)[: max(top_k, 1)]
        items: List[RetrievalItem] = []
        for i, idx in enumerate(top.tolist()):
            title, url, snippet = meta[idx]
            items.append(
                RetrievalItem(id=i, title=title, url=url, snippet=snippet, score=float(scores[idx]))
            )
        return items[:top_k]


def build_retriever(zim_path: str, kiwix_base: str) -> Retriever:
    return Retriever(zim_path, kiwix_base)
