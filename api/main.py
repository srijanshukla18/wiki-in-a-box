import asyncio
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse

from retrieval import Retriever, RetrievalItem, build_retriever


LLM_BASE = os.getenv("LLM_BASE", "http://ollama:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")
ZIM_PATH = os.getenv("ZIM_PATH", "/data/zims/enwiki.zim")
KIWIX_BASE = os.getenv("KIWIX_BASE", "http://kiwix:8080")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2700"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
OFFLINE_ENFORCE = os.getenv("OFFLINE_ENFORCE", "true").lower() == "true"
DEBUG_REASONING = os.getenv("DEBUG_REASONING", "false").lower() == "true"


retriever: Optional[Retriever] = None
 


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    try:
        if not os.path.exists(ZIM_PATH):
            print(f"[search] ZIM not found at {ZIM_PATH}", flush=True)
            retriever = None
        else:
            print(f"[search] using libzim search over {ZIM_PATH}", flush=True)
            retriever = build_retriever(ZIM_PATH, KIWIX_BASE)
    except Exception as e:
        print(f"[search] error during startup: {e}", flush=True)
        retriever = None
    yield


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wan_status() -> bool:
    try:
        with httpx.Client(timeout=1.0) as client:
            client.get("http://1.1.1.1", timeout=1.0)
        return True
    except Exception:
        return False


@app.get("/api/health")
def health():
    indices = {"zim_present": os.path.exists(ZIM_PATH)}
    return {
        "wan": _wan_status(),
        "model_name": LLM_MODEL,
        "indices": indices,
    }


# -------- Chat (streaming) --------


def build_system_prompt() -> str:
    return (
        "You are an offline assistant. Prefer to answer strictly from the provided CONTEXT. "
        "Use [1][2]-style citation markers that map to the CONTEXT list. Keep answers concise. "
        "If the CONTEXT lacks the information to answer the question, provide a concise best-effort explanation "
        "clearly prefixed with 'General (no local cite): ' and do not fabricate citations."
    )


def pack_context(items: List[RetrievalItem], max_tokens: int = MAX_CONTEXT_TOKENS) -> Tuple[str, List[Dict[str, Any]]]:
    # Simple token estimator: words
    def est_tokens(s: str) -> int:
        return max(1, len(s.split()))

    used = 0
    parts = []
    citations = []
    for i, it in enumerate(items, start=1):
        segment = f"[{i}] {it.title} — {it.snippet}\n"
        t = est_tokens(segment)
        if used + t > max_tokens:
            break
        used += t
        parts.append(segment)
        citations.append({
            "id": i,
            "title": it.title,
            # Public path via nginx proxy
            "url": f"/kiwix{it.url}",
            "snippet": it.snippet,
        })
    return "".join(parts), citations


async def stream_chat_completion(messages: List[Dict[str, Any]]) -> AsyncIterator[str]:
    url = f"{LLM_BASE}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        # Always low reasoning effort across the project
        "extra_body": {"reasoning_effort": "low"},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=60.0)) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise HTTPException(status_code=502, detail=f"LLM error: {text.decode(errors='ignore')}")
            yielded = False
            llm_error: Optional[str] = None
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # OpenAI stream frames: lines start with "data: ..."
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        # If nothing was produced, try a non-streaming fallback once
                        if not yielded:
                            try:
                                fallback = payload.copy()
                                fallback["stream"] = False
                                r = await client.post(url, json=fallback)
                                if r.status_code == 200:
                                    obj = r.json()
                                    content = obj.get("choices", [{}])[0].get("message", {}).get("content")
                                    if content:
                                        # emit as a single token event to keep SSE contract
                                        yield "event: token\n" + f"data: {json.dumps({'token': content})}\n\n"
                                    else:
                                        err = obj.get("error") or llm_error or "empty response"
                                        print(f"[llm] empty response: {err}", flush=True)
                                else:
                                    print(f"[llm] fallback status={r.status_code} body={r.text[:200]}", flush=True)
                            except Exception as e:
                                print(f"[llm] fallback error: {e}", flush=True)
                        yield "event: done\n" + "data: {}\n\n"
                        break
                    try:
                        obj = json.loads(data)
                        if isinstance(obj, dict) and obj.get("error"):
                            llm_error = str(obj.get("error"))
                        delta_obj = obj.get("choices", [{}])[0].get("delta", {})
                        if DEBUG_REASONING and isinstance(delta_obj, dict) and delta_obj.get("reasoning"):
                            yield "event: reasoning\n" + f"data: {json.dumps({'token': delta_obj.get('reasoning')})}\n\n"
                        delta = delta_obj.get("content")
                        if delta:
                            yielded = True
                            yield "event: token\n" + f"data: {json.dumps({'token': delta})}\n\n"
                    except json.JSONDecodeError:
                        # pass through raw
                        yield "event: raw\n" + f"data: {json.dumps({'raw': data})}\n\n"


@app.post("/api/chat")
async def chat(body: Dict[str, Any]):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    question: str = body.get("question", "").strip()
    top_k = int(body.get("k", 6))
    mode = body.get("mode", "default")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    items = retriever.search(question, top_k=top_k)
    context_text, citations = pack_context(items)

    system = {"role": "system", "content": build_system_prompt()}
    user_content = [
        f"CONTEXT:\n{context_text}",
        f"QUESTION: {question}",
    ]
    if mode == "eli10":
        user_content.append("Explain for a 10-year-old with short sentences.")
    elif mode == "advanced":
        user_content.append("Assume a college student; include equations if present.")
    elif mode == "quiz":
        user_content.append(
            'Generate N=5 MCQs as strict JSON: {"questions":[{"q","choices":[...],"answer","explanation","refs":[ids]}]} using only CONTEXT.'
        )
    user_content.append("Respond with [1][2]-style citations.")
    user = {"role": "user", "content": "\n".join(user_content)}
 

    messages = [system, user]

    async def event_gen() -> AsyncIterator[bytes]:
        # Send citations first so client can render chips
        pref = "event: citations\n" + f"data: {json.dumps({'citations': citations})}\n\n"
        yield pref.encode()
        async for chunk in stream_chat_completion(messages):
            yield chunk.encode()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/answer_from_page")
async def answer_from_page(body: Dict[str, Any]):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    path = str(body.get("path", "")).strip()
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    question: str = body.get("question", "").strip()
    top_k = int(body.get("k", 6))
    mode = body.get("mode", "default")

    # Build items from a single page
    items = retriever.search_in_path(path, question or "summarize", top_k=top_k)
    context_text, citations = pack_context(items)

    system = {"role": "system", "content": build_system_prompt()}
    user_content = [
        f"CONTEXT:\n{context_text}",
        f"QUESTION: {question or 'Summarize the page.'}",
    ]
    if mode == "eli10":
        user_content.append("Explain for a 10-year-old with short sentences.")
    elif mode == "advanced":
        user_content.append("Assume a college student; include equations if present.")
    elif mode == "quiz":
        user_content.append(
            'Generate N=5 MCQs as strict JSON: {"questions":[{"q","choices":[...],"answer","explanation","refs":[ids]}]} using only CONTEXT.'
        )
    user_content.append("Respond with [1][2]-style citations when using CONTEXT.")
    user = {"role": "user", "content": "\n".join(user_content)}

    messages = [system, user]

    async def event_gen() -> AsyncIterator[bytes]:
        pref = "event: citations\n" + f"data: {json.dumps({'citations': citations})}\n\n"
        yield pref.encode()
        async for chunk in stream_chat_completion(messages):
            yield chunk.encode()

    return StreamingResponse(event_gen(), media_type="text/event-stream")
