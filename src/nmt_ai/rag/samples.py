import hashlib
from pathlib import Path

import chromadb

from nmt_ai.models.invoice_draft import InvoiceDraft
from nmt_ai.models.vendor_invoices import Invoice
from nmt_ai.settings import SAMPLES_DB


def _make_doc_id(source_file: str | Path) -> str:
    canonical = str(Path(source_file).resolve()).lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _build_layout_doc(raw_text: str, invoice: Invoice) -> str:
    compact = " ".join(raw_text.split())
    header = compact[:1000]
    footer = compact[-1000:] if len(compact) > 1000 else ""
    return (
        f"Vendor: {(invoice.vendor_name or '').strip()} | "
        f"VendorNo: {(invoice.vendor_number or '').strip()} | "
        f"Customer: {(invoice.customer_name or '').strip()} | "
        f"Currency: {(invoice.currency or '').strip().upper()}\n"
        f"Header: {header}\n"
        f"Footer: {footer}"
    )


def commit_draft_file(
    json_path: Path | str,
    *,
    require_reviewed_flag: bool = True,
    verified_by: str = "manual_review",
    chroma_path: Path = SAMPLES_DB,
    collection_name: str = "SAMPLE_COLLECTION",
) -> str | None:
    """Load an edited InvoiceDraft JSON file, validate it, and upsert to ChromaDB."""
    draft = InvoiceDraft.load_json(json_path)

    if require_reviewed_flag and not draft.reviewed:
        print(f"Skipped {Path(json_path).name}: set 'reviewed': true after verification.")
        return None

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)

    resolved_pdf = Path(draft.source_file).resolve()
    doc_id = _make_doc_id(resolved_pdf)
    inv = draft.invoice

    metadata: dict[str, str | int | float | bool] = {
        "source_file": str(resolved_pdf),
        "file_name": resolved_pdf.name,
        "verified_by": verified_by,
        "vendor_name": (inv.vendor_name or "").strip(),
        "vendor_number": (inv.vendor_number or "").strip(),
        "customer_name": (inv.customer_name or "").strip(),
        "currency": (inv.currency or "").strip().upper(),
        "invoice_json": inv.model_dump_json(),
        "raw_text_excerpt": draft.raw_pdf_text[:4000],
    }

    collection.upsert(
        ids=[doc_id],
        documents=[_build_layout_doc(draft.raw_pdf_text, inv)],
        metadatas=[metadata],
    )
    return doc_id