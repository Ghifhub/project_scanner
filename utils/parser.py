"""
utils/parser.py
Turn raw OCR text lines from an Indonesian receipt (struk) into structured fields:
merchant, date, total, and line items. Tuned by regex + heuristics since layouts
vary a lot store to store.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ReceiptItem:
    name: str
    price: Optional[int] = None


@dataclass
class ParsedReceipt:
    merchant: Optional[str] = None
    date: Optional[str] = None  # normalized as YYYY-MM-DD when possible
    total: Optional[int] = None
    items: List[ReceiptItem] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)


TOTAL_KEYWORDS = [
    r"\btot", r"\btol", r"\bjumlah", r"\bgrand\s*tot",
    r"\bamount\s*(?:due|paid)", r"\bbalance\s*due", r"\bnet\s*payable",
]
# Lines that mention "total" but are NOT the final amount we want (subtotals, tax, etc.)
TOTAL_EXCLUDE_KEYWORDS = [
    r"sub\s*tot", r"diskon", r"discount", r"ppn", r"pajak", r"total\s*(?:item|qty|quantity)",
    r"jumlah\s*(?:item|qty|quantity)", r"hemat", r"saving", r"voucher",
]
GRAND_TOTAL_LABEL = re.compile(
    r"\b(?:grand\s*total|total(?:\s*(?:belanja|pembelian|tagihan|bayar|pembayaran|harga|akhir|due|paid|payable|payment|amount|purchase|transaction|bill))?|"
    r"jumlah(?:\s*(?:belanja|tagihan|bayar|pembayaran))?|amount(?:\s*(?:due|paid|payable))?|balance\s*due|net\s*(?:payable|amount))\b",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    # Year-first formats must come first: 2026-09-12 can also match day-first.
    r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)",
    # Day-first numeric dates with common separators and optional OCR spaces:
    # 19.10.26, 19. 10.26, 19/10/2026, 19 - 10 - 26, or 19,10,2026.
    r"(?<!\d)(\d{1,2})\s*[,./-]\s*(\d{1,2})\s*[,./-]\s*(\d{2,4})(?!\d)",
    # Some OCR engines omit punctuation entirely: 19 10 2026.
    r"(?<!\d)(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})(?!\d)",
]

MONTHS_ID = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "jun": 6,
    "jul": 7, "agu": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12,
    "may": 5, "aug": 8, "oct": 10, "dec": 12,
}


def _clean_number(s: str) -> Optional[int]:
    """Parse an Indonesian-formatted number ('45.000' or '45,000') into an int rupiah amount."""
    s = s.strip()
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    # Indonesian receipts may include cents as 24.500,00; ignore those decimals.
    if "," in s and re.search(r",\d{2}$", s) and "." in s:
        s = s.rsplit(",", 1)[0]
    # Indonesian convention: '.' or ',' as thousands separator, no decimals on receipts usually
    s = s.replace(".", "").replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


def _plausible_amount(s: str) -> bool:
    """Accept plain digits or thousands grouping, tolerating adjacent OCR punctuation."""
    normalized = re.sub(r"[.,]{2,}", ".", s)
    return normalized.isdigit() or bool(
        re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", normalized)
        or re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", normalized)
    )


def _amount_near_total_label(lines: List[str], index: int) -> Optional[int]:
    """Read a total beside its label, preferring same-row then nearest amount row."""
    amount_pattern = re.compile(r"(?<!\d)\d[\d.,]*(?:\s+\d{3})*(?!\d)")

    def parse_amounts(text: str) -> Optional[int]:
        text = re.sub(r"\s*([.,])\s*", r"\1", text)
        for number in reversed(amount_pattern.findall(text)):
            if _plausible_amount(number.replace(" ", "")):
                amount = _clean_number(number)
                if amount and amount > 100:
                    return amount
        return None

    same_row = parse_amounts(lines[index])
    if same_row is not None:
        return same_row

    # OCR may order a right-aligned total above or below the left-side label.
    # Use the closest numeric-only row, never a farther tender row.
    for offset in (1, -1, 2, -2):
        neighbor_index = index + offset
        if not 0 <= neighbor_index < len(lines):
            continue
        neighbor = re.sub(r"\b(?:rp|idr)\b", "", lines[neighbor_index], flags=re.IGNORECASE)
        neighbor = re.sub(r"\s*([.,])\s*", r"\1", neighbor).strip(" :|-")
        if re.fullmatch(r"[\d.,\s]+", neighbor):
            amount = parse_amounts(neighbor)
            if amount is not None:
                return amount
    return None


def extract_total(lines: List[str]) -> Optional[int]:
    """Find the receipt grand total across common Indonesian and English labels."""
    amount_pattern = re.compile(r"(?<!\d)\d[\d.,]*(?:\s+\d{3})*(?!\d)")

    # Explicit grand-total labels outrank tender amounts (cash/card), item prices,
    # and OCR's fallback guesses. The amount may be on the label row or the row below.
    for index, line in enumerate(lines):
        if not GRAND_TOTAL_LABEL.search(line) or any(re.search(kw, line, re.IGNORECASE) for kw in TOTAL_EXCLUDE_KEYWORDS):
            continue
        amount = _amount_near_total_label(lines, index)
        if amount is not None:
            return amount

    # If OCR drops the printed grand-total row, split tender rows still sum to
    # the amount paid (for example NON TUNAI 14,000 + TUNAI 6,000).
    tender_total = extract_total_from_split_tenders(lines)
    if tender_total is not None:
        return tender_total

    # On some Indonesian receipts OCR loses the "TOTAL BELANJA" label but
    # keeps the layout order: DISKON, then the grand total on the next row.
    for index, line in enumerate(lines):
        if not re.search(r"\b(?:diskon|discount|potongan)\b", line, re.IGNORECASE):
            continue
        for next_line in lines[index + 1:index + 3]:
            value_line = re.sub(r"\b(?:rp|idr)\b", "", next_line, flags=re.IGNORECASE).strip()
            value_line = value_line.strip(" :|-")
            if re.fullmatch(r"[\d.,]+", value_line):
                amount = _clean_number(value_line)
                if amount and amount > 100 and _plausible_amount(value_line):
                    return amount
            # Don't keep scanning past the next labeled row; that could be a
            # tender or a tax value rather than the grand total.
            if re.search(r"[a-zA-Z]", next_line):
                break

    candidates = []
    for line in lines:
        low = line.lower()
        if any(re.search(kw, low) for kw in TOTAL_EXCLUDE_KEYWORDS):
            continue
        if any(re.search(kw, low) for kw in TOTAL_KEYWORDS):
            normalized = re.sub(r"\s*([.,])\s*", r"\1", line)
            normalized = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", normalized)
            numbers = re.findall(r"(?<!\d)\d[\d.,]*(?:\s+\d{3})*(?!\d)", normalized)
            if numbers:
                amount = next((_clean_number(n) for n in reversed(numbers)
                               if _plausible_amount(n.replace(" ", ""))), None)
                if amount and amount > 100:
                    candidates.append(amount)
    if candidates:
        return candidates[-1]

    # Fallback: prefer plausible monetary values from the body/footer. Exclude
    # identifiers and contact details, which OCR often turns into large numbers.
    all_amounts = []
    for index, line in enumerate(lines):
        low = line.lower()
        if any(word in low for word in (
            "http", "www", "com", "link kritik", "saran", "wa ", "telp",
            "telepon", "phone", "customer", "layanan", "trxid", "trx id",
            "transaction id", "no. transaksi", "nomor transaksi", "no trans", "trans:",
            "transaksi", "harga jual", "npwp",
            "dpp", "ppn",
        )):
            continue
        # OCR may split a transaction label and its number into adjacent rows.
        if re.fullmatch(r"[\d.,\s]+", line.strip()):
            context = " ".join(lines[max(0, index - 2):index + 3]).lower()
            if re.search(r"\b(?:no\.?\s*trans|trans|transaksi|transaction)\b", context):
                continue
        if "/" in line and not any(re.search(pattern, line) for pattern in DATE_PATTERNS):
            continue
        # Date components (especially the year) are not receipt amounts.
        if any(re.search(pattern, line) for pattern in DATE_PATTERNS):
            continue
        for match in re.findall(r"[\d.,]{4,}(?:\s+\d{3})*", line):
            if not _plausible_amount(match):
                continue
            digits = re.sub(r"\D", "", match)
            # Unformatted 7+ digit strings are usually phone numbers, receipt
            # IDs, or OCR-merged fields. Real large amounts usually have grouping
            # separators or an explicit Rp/IDR currency marker.
            if len(digits) > 6 and not re.search(r"[.,]", match) and not re.search(r"\b(?:rp|idr)\b", line, re.IGNORECASE):
                continue
            amt = _clean_number(match)
            if amt and 100 < amt <= 1_000_000_000:
                all_amounts.append(amt)
    if not all_amounts:
        return None
    # OCR can duplicate item prices and payment amounts while missing the
    # "TOTAL BELANJA" label. In that case, the largest plausible receipt amount
    # is a safer fallback than the most frequently repeated amount.
    return max(all_amounts)


def score_total_candidate(lines: List[str], total: Optional[int]) -> int:
    """Rank OCR totals by how directly the text supports the parsed amount."""
    if total is None:
        return 0

    has_total_label = False
    for index, line in enumerate(lines):
        if GRAND_TOTAL_LABEL.search(line) and not any(
            re.search(keyword, line, re.IGNORECASE) for keyword in TOTAL_EXCLUDE_KEYWORDS
        ):
            has_total_label = True
            if _amount_near_total_label(lines, index) == total:
                return 100

    tender_total = extract_total_from_split_tenders(lines)
    if tender_total == total:
        return 80

    for index, line in enumerate(lines):
        if re.search(r"\b(?:diskon|discount|potongan)\b", line, re.IGNORECASE):
            for value_line in lines[index + 1:index + 3]:
                amount = _clean_number(value_line)
                if amount == total:
                    return 70
                if re.search(r"[a-zA-Z]", value_line):
                    break

    return 20 if has_total_label else 10


def extract_total_from_split_tenders(lines: List[str]) -> Optional[int]:
    """Sum explicitly labeled cash and non-cash portions of a split payment."""
    amount_pattern = re.compile(r"(?<!\d)\d[\d.,]*(?:\s+\d{3})*(?!\d)")
    non_cash_amounts = []
    cash_amounts = []
    for line in lines:
        low = line.lower()
        if not re.search(r"\b(?:non\s*tunai|tunai|cash)\b", low):
            continue
        numbers = amount_pattern.findall(line)
        amount = next((_clean_number(number) for number in reversed(numbers)
                       if _plausible_amount(number.replace(" ", ""))), None)
        if not amount or amount <= 100:
            continue
        if re.search(r"\bnon\s*tunai\b", low):
            non_cash_amounts.append(amount)
        else:
            cash_amounts.append(amount)
    if non_cash_amounts and cash_amounts:
        return sum(non_cash_amounts) + sum(cash_amounts)
    return None


def extract_date(lines: List[str]) -> Optional[str]:
    for line in lines:
        for pattern in DATE_PATTERNS:
            m = re.search(pattern, line)
            if m:
                parts = m.groups()
                try:
                    if len(parts[0]) == 4:  # YYYY-MM-DD form
                        y, mo, d = parts
                    else:  # DD-MM-YYYY form
                        d, mo, y = parts
                        if len(y) == 2:
                            y = "20" + y
                    dt = datetime(int(y), int(mo), int(d))
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        # Support both "Oct 17 2026" and the common receipt form "17 Oct 2026".
        month_names = r"jan|feb|mar|apr|may|mei|jun|jul|agu|aug|sep|oct|okt|nov|dec|des"
        month_match = re.search(
            rf"\b({month_names})\.?\s+(\d{{1,2}}),?\s+(\d{{2,4}})\b"
            rf"|\b(\d{{1,2}})\s+({month_names})\.?[,]?\s+(\d{{2,4}})\b",
            line,
            re.IGNORECASE,
        )
        if month_match:
            if month_match.group(1):
                month, day, year = month_match.group(1, 2, 3)
            else:
                day, month, year = month_match.group(4, 5, 6)
            month_number = MONTHS_ID.get(month[:3].lower())
            if month_number:
                if len(year) == 2:
                    year = "20" + year
                try:
                    return datetime(int(year), month_number, int(day)).strftime("%Y-%m-%d")
                except ValueError:
                    pass
    return None


def extract_merchant(lines: List[str]) -> Optional[str]:
    """Heuristic: merchant name is usually one of the first few non-empty,
    mostly-alphabetic lines at the top of the receipt."""
    for line in lines[:5]:
        cleaned = line.strip()
        if re.search(r"\bbeepos\b", cleaned, re.IGNORECASE):
            continue
        if len(cleaned) < 5:
            continue
        letters = sum(c.isalpha() for c in cleaned)
        if letters / max(len(cleaned), 1) > 0.5:
            return cleaned
    return lines[0].strip() if lines else None


def extract_items(lines: List[str]) -> List[ReceiptItem]:
    """
    Extract item names and prices from common receipt layouts. Names and prices
    may share a line, appear on adjacent lines, or be printed in price-first order.
    """
    items = []
    skip_keywords = TOTAL_KEYWORDS + TOTAL_EXCLUDE_KEYWORDS + [
        r"tota", r"bayar", r"tunai", r"kembali", r"kembal\s*i?an", r"cash", r"change", r"kasir",
        r"npwp", r"terima\s*kasih", r"http", r"www", r"com", r"link kritik", r"saran",
        r"subtotal", r"payment", r"paynent", r"check\s*no", r"tanggal", r"transaksi",
        r"operator", r"pelanggan", r"alamat", r"member", r"channel", r"shift", r"\bbal\b", r"\bnama\b",
        r"produk", r"volume", r"harga/liter", r"pertamina", r"supported by",
    ]
    amount_pattern = re.compile(r"(?<!\d)\d[\d.,]*(?:\s+\d{3})*(?!\d)")
    pending_name = None

    for line in lines:
        # OCR often inserts spaces around the thousands separator.
        normalized = re.sub(r"\s*([.,])\s*", r"\1", line).strip()
        low = normalized.lower()
        if any(re.search(kw, low) for kw in skip_keywords):
            pending_name = None
            continue

        matches = [m for m in amount_pattern.finditer(normalized)
                   if _plausible_amount(m.group().replace(" ", ""))]
        if not matches:
            # Keep a likely item label for receipts that print its price below.
            if any(char.isalpha() for char in normalized) and not re.search(r"\d", normalized):
                pending_name = normalized.strip(" :-")
            continue

        # A leading price followed by a product name is used by some receipts.
        first = matches[0]
        prefix, suffix = normalized[:first.start()], normalized[first.end():]
        if not prefix.strip(" :Rp") and any(char.isalpha() for char in suffix):
            name = suffix.strip(" :-")
            raw_price = first.group()
        else:
            last = matches[-1]
            raw_price = last.group()
            prefix = normalized[:last.start()]
            # Ignore quantity and unit price when a line ends with the extended price.
            quantity = re.search(r"\b\d*\s*[xX]\s*", prefix)
            if quantity:
                prefix = prefix[:quantity.start()]
            name = re.sub(r"\bRp\b", "", prefix, flags=re.IGNORECASE).strip(" :-xX0123456789.,")
            if not any(char.isalpha() for char in name):
                name = pending_name or ""

        price = _clean_number(raw_price)
        if name and price and price > 50:
            items.append(ReceiptItem(name=name, price=price))
        pending_name = None
    return items


def parse_receipt(lines: List[str]) -> ParsedReceipt:
    """Run all extractors over OCR'd text lines and return a structured receipt."""
    return ParsedReceipt(
        merchant=extract_merchant(lines),
        date=extract_date(lines),
        total=extract_total(lines),
        items=extract_items(lines),
        raw_lines=lines,
    )


