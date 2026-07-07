from pathlib import Path

from nmt_ai.pre_process.pdf_extract import PDFExtract
from nmt_ai.settings import *
from nmt_ai.pydantic_ai4 import check_invoice
invoices_folder_str = DATA_PATH + "/invoices"
invoices_folder_path = Path(invoices_folder_str)

pdf_files = [p for p in invoices_folder_path.rglob("*") if p.suffix.lower() == ".pdf"]
# for f in pdf_files:
#     print(f.absolute())


for file in pdf_files:
    f = file
    input_file = invoices_folder_str + "/" + file.name

    e = PDFExtract()
    print(input_file)
    text = e.extract(input_file)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    print(text)
    check_invoice(text)

