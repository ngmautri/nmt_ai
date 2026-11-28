import time
from pathlib import Path

from nmt_ai.models.invoice_draft import InvoiceDraft
from nmt_ai.pre_process.pdf_extract import PDFExtract
from colorama import Fore, Style

# input_data =get_input()
from nmt_ai.pre_process.pdf_extract import PDFExtract
from nmt_ai.pydantic_ai6 import check_invoice
from nmt_ai.models.vendor_invoices import InvoiceOutput
from nmt_ai.rag.samples import commit_draft_file
from nmt_ai.settings import *


reviewed_folder = DATA_PATH + "/samples_reviewed"
reviewed_folder_path = Path(reviewed_folder)

def create_folder (msg_id):
    print(len(msg_id))
    level1 = msg_id[:len(msg_id)-16]
    level2_tmp = msg_id[-16:]
    level2 = level2_tmp[:11]
    level3 = level2_tmp[-5:]

    return (f'{level1}\\{level2}\\{level3}')


# pdffiles = [f for f in listdir(input_path) if isfile(join(input_path, f)) and '.PDF' in f]

reviewed_files = [p for p in reviewed_folder_path.glob("*") if p.suffix.lower() == ".json"]

print('\nList of JOSN Files:\n')
for review_f in reviewed_files:
    print(review_f)
    draft_file = InvoiceDraft.load_json(review_f)

    # Step 2: Open `draft_file` in VS Code / editor, fix any `invoice` fields,
    # and change `"reviewed": false` -> `"reviewed": true`.

    # Step 3: Reload and commit verified drafts to ChromaDB
    doc_id = commit_draft_file(review_f)
    if doc_id:
        print(f"Committed verified invoice to ChromaDB: {doc_id}")



