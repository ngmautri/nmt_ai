import time
from pathlib import Path

from nmt_ai.pre_process.pdf_extract import PDFExtract
from nmt_ai.pydantic_ai4 import check_invoice
from nmt_ai.models.vendor_invoices import InvoiceOutput
from nmt_ai.settings import *
invoices_folder_str = DATA_PATH + "/invoices"
invoices_folder_path = Path(invoices_folder_str)

pdf_files = [p for p in invoices_folder_path.glob("*") if p.suffix.lower() == ".pdf"]
# for f in pdf_files:
#     print(f.absolute())

result_list =[]

print(f">>>>>>> Please wait.....")
start = time.time()

for file in pdf_files:
    f = file
    input_file = invoices_folder_str + "\\" + file.name
    print(input_file)
    e = PDFExtract()

    text = e.extract(input_file)
    # text = text.replace("\r", " ")
    # text = text.replace("\n", " ")
    # print(text)
    r = check_invoice(text,input_file)

    if len(r) > 0:
        for i in r:
            result_list.append(i)

output_path = DATA_PATH + "\\invoices" + "\\output"
args = {"result_list": result_list, "output_path": output_path}
output = InvoiceOutput()
output.export_to_excel(args)

# print(result_list)
end = time.time()
print(f"=>>>>>> Execution time: {end - start:.6f} seconds")
print(f"---------------------------------------------------")


