from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from pydantic import BaseModel, Field

from nmt_ai.models.vendor_invoices import Invoice
from nmt_ai.settings import SAMPLES_DB

CHROMA_PATH = Path(SAMPLES_DB)
COLLECTION_NAME = "verified_invoices"


class InvoiceDraft(BaseModel):
    source_file: str
    raw_pdf_text: str
    invoice: Invoice
    reviewed: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save_json(self, output_path: Path | str) -> Path:
        """Serialize the draft to a pretty-printed JSON file for manual review."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, json_path: Path | str) -> "InvoiceDraft":
        """Reload and validate an edited draft JSON file from disk."""
        raw_json = Path(json_path).read_text(encoding="utf-8")
        return cls.model_validate_json(raw_json)