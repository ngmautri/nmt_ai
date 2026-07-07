from __future__ import annotations

import re
from abc import ABC, abstractmethod

import fitz  # imports the pymupdf library
import pdfplumber
import pypdf
from PIL import Image
from pytesseract import pytesseract
from tika import parser


