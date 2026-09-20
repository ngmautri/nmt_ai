import asyncio
import time
import traceback

import pydantic_ai
from openpyxl import Workbook
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field, ValidationError, field_validator

from typing import List, Optional, Literal
from nmt_ai.helpers import const
import numpy as np


class LineItem(BaseModel):
    description: Optional[str]
    quantity: Optional[float]
    unit_price: Optional[float]
    amount: Optional[float]


class Invoice_1(BaseModel):
    invoice_number: Optional[str] = Field(description="invoice number")
    # invoice_date: Optional[str]
    invoice_date: Optional[str] = Field(description="Invoice Date")
    language: Optional[str] = Field(default=None, description="the language of invoice")

    # due_date: Optional[str]
    due_date: Optional[str] = Field(description="the due date of invoice")
    vendor_name: Optional[str] = Field(description="vendor name")

    # it does not work with vendor_vat_number
    vendor_vat: Optional[str] = Field(default=None, description="Vendor VAT number")

    # vendor_number: Optional[str]  =  Field(description="vendor number")

    customer_name: Optional[str] = Field(description="customer name. e.g Mascot")
    currency: Optional[str] = Field(description="currency")
    payment_terms: Optional[str] = Field(description="payment term")

    subtotal: Optional[float] = Field(description="Pre-tax invoice total — sum of line items before VAT")
    tax_rate: Optional[float] = Field(description="Tax rate")
    tax_amount: Optional[float] = Field(description="Total VAT charged on the invoice")
    total: Optional[float] = Field(description="Final invoice total — net_total plus vat_amount")
    # confidence: float = Field(default=0.1 ,description="Confidence score of the extraction between 0.0 and 1.0.", ge=0.0, le=1.0)
    confidence: float = Field(default=0.1, description="Confidence score of the extraction between 0.0 and 1.0.")


class Invoice2(BaseModel):
    invoice_number: Optional[str] = Field(default=None, description="invoice number")
    invoice_date: Optional[str] = Field(default=None, description="Invoice Date")
    language: Optional[str] = Field(default=None, description="the language of invoice")
    due_date: Optional[str] = Field(default=None, description="the due date of invoice")
    vendor_vat: Optional[str] = Field(default=None, description="Vendor VAT number")
    customer_name: Optional[str] = Field(default=None, description="customer name. e.g Mascot")
    currency: Optional[str] = Field(default=None, description="currency")
    payment_terms: Optional[str] = Field(default=None, description="payment term")

    subtotal: Optional[float] = Field(default=None, description="Pre-tax invoice total — sum of line items before VAT")
    tax_rate: Optional[float] = Field(default=None, description="Tax rate")
    tax_amount: Optional[float] = Field(default=None, description="Total VAT charged on the invoice")
    total: Optional[float] = Field(default=None, description="Final invoice total — net_total plus vat_amount")

    # XÓA ge=0.0 và le=1.0 ở đây
    confidence: float = Field(default=0.1, description="Confidence score of the extraction between 0.0 and 1.0.")

    # Thay thế bằng validator của Pydantic để không làm lỗi Ollama Grammar
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class Invoice_tmp1(BaseModel):
    """
      A structured representation of an invoice document.

      This Pydantic model defines the key attributes extracted from invoices,
      including metadata (invoice number, dates, language), vendor and customer
      details, financial amounts, and confidence scores. It is designed to support
      automated invoice parsing, validation, and integration with downstream
      systems such as Granite 4.2 for schema-guided generation.

      Fields:
          invoice_number (Optional[str]): Unique identifier for the invoice.
          invoice_date (Optional[str]): Date the invoice was issued.
          language (Optional[str]): Language of the invoice text.
          due_date (Optional[str]): Payment due date.
          vendor_name (Optional[str]): Name of the vendor issuing the invoice.
          vendor_vat (Optional[str]): Vendor's VAT number.
          customer_name (Optional[str]): Name of the customer (e.g., Mascot).
          currency (Optional[str]): Currency code (e.g., USD, EUR).
          payment_terms (Optional[str]): Payment terms specified on the invoice.
          subtotal (Optional[float]): Pre-tax total of line items.
          tax_rate (Optional[float]): Applied tax rate.
          tax_amount (Optional[float]): Total VAT charged.
          total (Optional[float]): Final invoice total (subtotal + tax).
          confidence (float): Confidence score of extraction (0.0–1.0).
      """
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date")
    language: Optional[str] = Field(None, description="Language of invoice")
    due_date: Optional[str] = Field(None, description="Due date")
    vendor_name: Optional[str] = Field(description="Vendor name")
    vendor_vat: Optional[str] = Field(None, description="Vendor VAT number")
    vendor_number: Optional[str] = Field(None, description="Vendor number in SAP to be retrieved from Tool.")
    vendor_name_sap: Optional[str] = Field(None, description="Vendor number in SAP to be retrieved from Tool.")
    customer_name: Optional[str] = Field(None, description="Customer name (must contain Mascot)")
    currency: Optional[Literal["USD", "EUR", "VND", "DKK", "SEK", "NOK", "PLN"]] = Field(None,
                                                                                         description="Currency code")
    payment_terms: Optional[str] = Field(None, description="Payment terms")

    subtotal: Optional[float] = Field(None, description="Pre-tax total")
    tax_rate: Optional[float] = Field(None, description="Tax rate")
    tax_amount: Optional[float] = Field(None, description="VAT amount")
    total: Optional[float] = Field(None, description="Final total")

    confidence: float = Field(0.0, description="Confidence score between 0.0 and 1.0")

    @field_validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @field_validator("customer_name")
    def validate_customer(cls, v: Optional[str]) -> Optional[str]:
        if v and "mascot" not in v.lower():
            return None
        return v

    # @field_validator("vendor_name")
    # def validate_vendor(cls, v: Optional[str]) -> Optional[str]:
    #     if v and "mascot" in v.lower():
    #         return None
    #     return


class Invoice(BaseModel):
    """
    A structured representation of an invoice document.
    """

    invoice_number: Optional[str] = Field(
        default=None,
        description="Unique identifier assigned to the invoice (e.g., INV-2026-001)."
    )
    invoice_date: Optional[str] = Field(
        default=None,
        description="Date the invoice was issued, ideally in ISO 8601 format (YYYY-MM-DD)."
    )
    language: Optional[str] = Field(
        default=None,
        description="Language in which the invoice is written (e.g., 'EN', 'FR')."
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Date by which payment must be made, in ISO 8601 format."
    )
    vendor_name: Optional[str] = Field(
        default=None,
        description="Legal or trading name of the vendor issuing the invoice.")

    vendor_name_sap: Optional[str] = Field(
        default=None,
        description="name of the vendor in SAP to be retrieved from Tool."
    )
    vendor_vat: Optional[str] = Field(
        default=None,
        description="Vendor's VAT (Value Added Tax) registration number."
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Name of the customer or client receiving the invoice."
    )

    customer_address: Optional[str] = Field(
        default=None,
        description="address of the customer or client of the invoice."
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code for monetary values, following ISO 4217 (e.g., 'USD', 'EUR')."
    )
    payment_terms: Optional[str] = Field(
        default=None,
        description="Agreed payment terms (e.g., 'Net 30', 'Due on receipt')."
    )
    subtotal: Optional[float] = Field(
        default=None,
        description="Total amount before tax, calculated as the sum of all line items."
    )
    tax_rate: Optional[float] = Field(
        default=None,
        description="Applied tax rate as a percentage (e.g., 10.0 for 10%)."
    )
    tax_amount: Optional[float] = Field(
        default=0,
        description="Total tax charged on the invoice, based on subtotal and tax rate."
    )
    total: Optional[float] = Field(
        default=None,
        description="Final payable amount, equal to subtotal plus tax_amount."
    )
    confidence: float = Field(
        default=0.1,
        description="Confidence score (0.0–1.0) indicating reliability of extracted values.",
        ge=0.0,
        le=1.0
    )


class InvoiceOutput:
    def export_to_excel(self, *kwargs):
        try:
            # print(kwargs)
            if type(kwargs) != tuple:
                raise Exception("Sorry, the done is empty1")
                return

            if len(kwargs[0]) == 0:
                raise Exception("Sorry, the done is empty2")
                return

            result_list = kwargs[0]['result_list']
            output_path = kwargs[0]['output_path']

            workbook = Workbook()
            sheet = workbook.active

            headers = np.array([
                "#",
                "vendor_name",
                "vendor_name_sap",
                "vendor_vat",
                "invoice_number",
                "invoice_date",
                "due_date",
                "language",
                "customername",
                "customer address",
                "currency",
                "subtotal",
                "tax_rate",
                "tax_amount",
                "total",
                "pmt term",
                "source",
                "model",
                "confidence",
                "duration",
            ])

            n = 0
            header_row = 2
            col_no = 1
            for h in headers:
                excel_col = const.excel_cols[col_no]
                cell = excel_col + str(header_row)
                sheet[cell] = h
                col_no = col_no + 1

            counter = 1

            for i in result_list:
                print(i)
                llm_model = i['llm_model']
                input_file = i['input_file']
                duration = i['duration']
                r = i["invoice"]
                if (isinstance(r, Invoice)):
                    cols = np.array([
                        counter,
                        r.vendor_name,
                        r.vendor_name_sap,
                        r.vendor_vat,
                        r.invoice_number,
                        r.invoice_date,
                        r.due_date,
                        r.language,
                        r.customer_name,
                        r.customer_address,
                        r.currency,
                        r.subtotal,
                        r.tax_rate,
                        r.tax_amount,
                        r.total,
                        r.payment_terms,
                        input_file, llm_model, r.confidence, duration
                    ])

                    col_no = 1

                    for c in cols:
                        excel_col = const.excel_cols[col_no]
                        cell = excel_col + str(header_row + counter)
                        sheet[cell] = c
                        col_no = col_no + 1
                    counter = counter + 1

            workbook.save(filename=output_path + "\\output.xlsx")

        except Exception as e:
            print(e)
            traceback.print_exc()
