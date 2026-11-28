# --- ChromaDB RAG Retrieval ---
import asyncio
import json
from pathlib import Path
from typing import Any

import chromadb

from nmt_ai.settings import SAMPLES_DB


def get_rag_collection(
    chroma_path: Path = SAMPLES_DB,
    collection_name: str = "SAMPLE_COLLECTION",
):
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(name=collection_name)


def _build_query_snippet(raw_text: str) -> str:
    compact = " ".join(raw_text.split())
    head = compact[:1000]
    tail = compact[-1000:] if len(compact) > 1000 else ""
    return f"Header: {head}\nFooter: {tail}"


def _parse_rag_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    metadatas_batch = results.get("metadatas") or []
    distances_batch = results.get("distances") or []
    if not metadatas_batch or not metadatas_batch[0]:
        return []

    parsed: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadatas_batch[0]):
        if not meta:
            continue
        raw_json = meta.get("invoice_json", "{}")
        parsed.append({
            "source_file": meta.get("source_file", ""),
            "distance": distances_batch[0][idx] if distances_batch and distances_batch[0] else None,
            "raw_text": meta.get("raw_text_excerpt", ""),
            "extracted_invoice": json.loads(raw_json) if isinstance(raw_json, str) else {},
        })
    return parsed


async def retrieve_similar_invoices(
    raw_pdf_text: str,
    *,
    top_k: int = 2,
    currency: str | None = None,
    collection: Any | None = None,
) -> list[dict[str, Any]]:
    col = collection or get_rag_collection()
    if col.count() == 0:
        return []

    query_text = _build_query_snippet(raw_pdf_text)
    where_filter = {"currency": currency.strip().upper()} if currency else None

    results = await asyncio.to_thread(
        col.query,
        query_texts=[query_text],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"],
    )
    samples = _parse_rag_results(results)

    # Fallback without currency filter if filtered query returns fewer than top_k
    if len(samples) < top_k and where_filter is not None:
        fallback = await asyncio.to_thread(
            col.query,
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        seen = {s["source_file"] for s in samples}
        for item in _parse_rag_results(fallback):
            if item["source_file"] not in seen:
                samples.append(item)
                seen.add(item["source_file"])
            if len(samples) >= top_k:
                break

    return samples