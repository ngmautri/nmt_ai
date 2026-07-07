from __future__ import annotations

import re
from abc import ABC, abstractmethod

import pymupdf  # imports the pymupdf library
import pdfplumber
import pypdf
from PIL import Image
from pytesseract import pytesseract
from tika import parser


class PDFExtractStategy(ABC):
    @abstractmethod
    def extract(self, input: str)->str:
        pass



class PDFExtract(PDFExtractStategy):
    def extract(self, input: str)->str:
        try:
            reader = pypdf.PdfReader(input)
            text = ""
            print (f"No of Pages: {len(reader.pages)}")
            for page in reader.pages:
               text += page.extract_text() + "\n"

            return text
        except:
            return None


# class FitxExtract(PDFExtractStategy):
#     def extract(self, input: str)->str:
#         text = ""
#         doc = fitz.open(input)  # open a document
#
#         print(f"Page No: {len(doc)}")
#
#         for page in doc:  # iterate the document pages
#             text += page.get_text()  # get plain text encoded as UTF-8
#
#         return text

class PDFPflumberExtract(PDFExtractStategy):
    def extract(self, input: str)->str:
        text = ""

        all_text = []
        with pdfplumber.open(input) as pdf:

            print(f"Page No: {len(pdf.pages)}")

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)

        text = "\n".join(all_text)

        return text


class TikaExtract(PDFExtractStategy):
    def extract(self, input: str)->str:
        raw = parser.from_file(input)
        raw = str(raw)

        safe_text = raw.encode('utf-8', errors='ignore')

        safe_text = str(safe_text).replace("\n", "").replace("\\", "")
        print('--- safe text ---')
        return (safe_text)


# class TesseractExtract(PDFExtractStategy):
#     def extract(self, input: str)->str:
#         path_to_tesseract = 'C:\\Users\\nmt\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'  # Define path to tessaract.exe
#         pytesseract.tesseract_cmd = path_to_tesseract  # Point tessaract_cmd to tessaract.exe
#
#         image = Image.open(input)  # Open image with Pillow
#         text = pytesseract.image_to_string(image)  # Extract text from image
#         return text

