import os
import sqlite3
from typing import Generator, Iterable, List, Tuple

from libzim.reader import Archive


def _db_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "titles.sqlite")


def build_title_index(zim_path: str, out_dir: str, max_rows: int | None = None) -> Generator[Tuple[float, str], None, None]:
    """Build a lightweight SQLite FTS5 index of article titles → paths.

    - Creates FTS5 virtual table titles(title, path)
    - Iterates entries in the ZIM, adds rows for textual articles only
    - Emits progress as (fraction, message)
    """
    dbp = _db_path(out_dir)
    if os.path.exists(dbp):
        os.remove(dbp)
    conn = sqlite3.connect(dbp)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("CREATE VIRTUAL TABLE titles USING fts5(title, path, tokenize = 'porter')")

    z = Archive(zim_path)
    total = int(getattr(z, "all_entry_count", 0)) or 0
    if max_rows:
        total = min(total, max_rows)
    batch = []
    written = 0
    def flush():
        nonlocal written
        if batch:
            conn.executemany("INSERT INTO titles(title, path) VALUES (?, ?)", batch)
            conn.commit()
            written += len(batch)
            batch.clear()
    for i in range(total):
        try:
            entry = z._get_entry_by_id(i)
            item = entry.get_item()
            mt = (item.mimetype or "").lower()
            if not (mt.startswith("text/") or "html" in mt):
                continue
            title = entry.title or ""
            if not title:
                continue
            path = entry.path
            batch.append((title, path))
            if len(batch) >= 1000:
                flush()
                frac = (i + 1) / max(1, total)
                yield (frac, f"indexed {written} titles")
        except Exception:
            continue
    flush()
    conn.close()
    yield (1.0, f"done, titles={written}")


def title_db_path(out_dir: str) -> str:
    return _db_path(out_dir)


def search_titles(out_dir: str, query: str, limit: int = 20) -> List[Tuple[str, str]]:
    """Search the FTS5 title index. Returns list of (path, title).

    Note: We rely on FTS5 default ranking. For custom ranking, a UDF over matchinfo()
          can be added later. In practice, an OR-joined token query works well.
    """
    dbp = _db_path(out_dir)
    if not os.path.exists(dbp):
        return []
    conn = sqlite3.connect(dbp)
    q = "SELECT path, title FROM titles WHERE titles MATCH ? LIMIT ?"
    try:
        rows = conn.execute(q, (query, int(limit))).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        return []


if __name__ == "__main__":
    import os as _os
    import sys as _sys
    zim = _os.environ.get("ZIM_PATH", "/data/zims/enwiki.zim")
    out = _os.environ.get("TITLE_INDEX_DIR", "/data/title_index")
    limit_env = _os.environ.get("LIMIT")
    max_rows = int(limit_env) if limit_env else None
    print(f"[title-index] building from {zim} -> {out} limit={max_rows}")
    try:
        for frac, msg in build_title_index(zim, out, max_rows=max_rows):
            print(f"[title-index] {frac:.1%} {msg}", flush=True)
    except KeyboardInterrupt:
        print("[title-index] interrupted", file=_sys.stderr)
