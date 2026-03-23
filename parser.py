#!/usr/bin/env python3
"""
Extract parameter/value/unit triples from DCF Dual Beam–style log PDFs
into a tabular CSV using pypdf (v6.8).

Highlights
----------
- Uses pypdf.PdfReader to read page text.
- Parses sequences like: "MV S2-1 DC 5.00 sccm".
- Supports string values as well (e.g., "PolSR TORVIS", "ModeSR gvm", "PosSR out", "AtomNumb He").
- Captures an optional unit immediately after the value (e.g., kV, mA, %, AMU, Pa/Ta).
- Tracks file name, page number, and section header context.
- CLI accepts a single PDF or a directory (recursive).

Usage
-----
  python extract_log_pdf_table_pypdf.py input.pdf -o output.csv
  python extract_log_pdf_table_pypdf.py /path/to/folder -o all_logs.csv
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
from pypdf import PdfReader


# --- Configuration you can tweak ------------------------------------------------

# Section headers seen in these logs; extend if you encounter new ones.
SECTION_HEADERS = [
    "Source 2",
    "Source 1",
    "Pre Acceleration",
    "Accelerator",
    "Post Acceleration",
    "-15 Degree Beamline",
    "Faraday Cups",
    "Vacuum",
    "Machine Setup",
]

# Common units found in the logs; extend as needed (e.g., add 'mbar', 'Hz', 's', etc.).
UNITS = {
    "sccm", "A", "V", "kV", "mA", "uA", "C", "G", "T", "mm", "psig", "DegC",
    "MV", "%", "AMU", "Pa/Ta", "u"
}

# Optional: helps skip non-data prefixes faster; not strictly required.
LIKELY_PARAM_STARTS = {
    "MV","FIL","ARC","EXT","FOC","GAP","BIA","OVN","CHM","INJ","IGC","ES","EL",
    "IM","TNK","CH","COL","CPS","TPS","GS","MQ","MS","SM","HPB","SS","IP","CVG","FC",
    "SETUP"
}

# Literal value words we’ve seen
VALUE_LITERALS = {"gvm", "out", "in", "TORVIS", "He"}  # extend as needed

# Tokens that typically PRECEDE a literal (non-numeric) value.
# Many of these end with 'SR'; generalize with endswith('SR').
LITERAL_VALUE_KEYS = {"AtomNumb", "SrcSel", "BLsel"}  # explicit list; 'SR' keys handled by suffix rule

INDEX_TOKEN_RE = re.compile(r"^[+-]?\d+(?:-\d+)?$")  # -1, 1, 01-1, 2-3, etc.

# --- Helpers --------------------------------------------------------------------

NUM_RE = re.compile(
    r"""
    ^[+-]?(
        (?:\d+\.\d*|\.\d+|\d+)       # int or float
        (?:[eE][+-]?\d+)?            # optional exponent
    )$
    """,
    re.VERBOSE,
)

def is_number(tok: str) -> bool:
    return bool(NUM_RE.match(tok))

def looks_like_unit(tok: str) -> bool:
    # Quick accept if in known set
    if tok in UNITS:
        return True
    # Otherwise heuristic: short-ish, letters/symbols, not a pure number
    if is_number(tok):
        return False
    return bool(re.match(r"^[A-Za-z%°/][A-Za-z0-9%°/\.\-\+]*$", tok))

def is_probable_param_start(tok: str) -> bool:
    if tok in LIKELY_PARAM_STARTS:
        return True
    # Many tags are uppercase 2–5 letters, optionally followed by digits/hyphens
    return bool(re.match(r"^[A-Z]{2,5}[A-Z0-9\-]*$", tok))


def expects_literal_after(last_param_token: str) -> bool:
    """
    Parameters that expect a non-numeric literal value immediately after them.
    Covers:
      - ...SR keys (ModeSR, PolSR, PosSR, etc.)
      - ...Sel keys (SrcSel, BLsel, etc.)
      - any explicitly listed in LITERAL_VALUE_KEYS
    """
    return (
        last_param_token.endswith("SR")
        or last_param_token.endswith("Sel")
        or last_param_token in LITERAL_VALUE_KEYS
    )


def is_literal_value_given_context(tok: str, param_tokens: list[str]) -> bool:
    """
    Accept a token as a literal value if:
      - it is explicitly listed (e.g., TORVIS, gvm, out, He), OR
      - the preceding param token expects a literal, and this token is not a number
        and not a new parameter start (unit-like appearance is allowed here).
    """
    if tok in VALUE_LITERALS:
        return True
    if param_tokens and expects_literal_after(param_tokens[-1]):
        # IMPORTANT: do NOT reject on looks_like_unit() here,
        # otherwise codes like S2/L3 will be blocked.
        if not is_number(tok) and not is_probable_param_start(tok):
            return True
    return False


def is_index_token(tok: str) -> bool:
    # Numeric tokens that are often part of a parameter path, not the value
    return bool(INDEX_TOKEN_RE.match(tok))

def is_suffix_tag(tok: str) -> bool:
    """
    Tags that can follow an index and are part of the parameter:
    examples: PR, CR, VC, VR, DC, DR, +VR, -VR, YVC, XVC, LastCR, TotInjV, etc.
    Heuristics:
      - starts with '+' or '-' followed by letters/digits (e.g., +VR, -VR), OR
      - looks like a short-ish tag beginning with a capital letter (PR, VC, YVC, LastCR, TotInjV).
    """
    if tok.startswith(('+', '-')) and re.match(r"^[\+\-][A-Za-z][A-Za-z0-9]*$", tok):
        return True
    # Permit CamelCase and all-caps tags commonly seen in these logs
    return bool(re.match(r"^[A-Z][A-Za-z0-9]{1,10}$", tok))

def find_section_in_line(line: str) -> Tuple[Optional[str], str]:
    """
    If a known section header appears in the line, return (header, line_without_that_header_once).
    Otherwise, (None, line).
    """
    for header in SECTION_HEADERS:
        if header in line:
            return header, line.replace(header, " ", 1)
    return None, line


def parse_line_into_rows(
    line: str, file_name: str, page_num: int, current_section: Optional[str]
) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse one line into zero or more rows of (parameter, value, unit).
    Updates current_section if a section header is found.
    """
    rows: List[Dict] = []

    # Detect & consume section header (if present)
    section, work_line = find_section_in_line(line)
    if section:
        current_section = section

    # Skip document header lines entirely
    if re.search(r"\bDCF\s+Dual\s+Beam\b", work_line, flags=re.IGNORECASE):
        return [], current_section

    tokens = work_line.strip().split()
    if not tokens:
        return rows, current_section

    i = 0
    n = len(tokens)

    # Skip obvious non-data prefix until something that looks like a parameter start
    while i < n and not is_probable_param_start(tokens[i]):
        i += 1

    while i < n:
        # Require parameter to begin at a plausible tag; otherwise advance
        if not is_probable_param_start(tokens[i]):
            i += 1
            continue

        # Accumulate parameter tokens until we hit a VALUE:
        #   - numeric value (always), OR
        #   - literal value ONLY when the preceding param token expects a literal
        param_tokens: List[str] = []
        while i < n:
            tok = tokens[i]
            nxt = tokens[i + 1] if i + 1 < n else ""

            # 1) If tok is numeric and NOT an index that's followed by a tag,
            #    then it's the VALUE → stop accumulating
            if is_number(tok) and not (is_index_token(tok) and is_suffix_tag(nxt)):
                break

            # 2) If tok is an allowed literal VALUE given the current context (e.g., after ModeSR/PolSR/PosSR/AtomNumb)
            if is_literal_value_given_context(tok, param_tokens):
                break

            # Otherwise, still part of the parameter
            param_tokens.append(tok)
            i += 1

            if i >= n:
                # No value token found on this line
                return rows, current_section

        if i >= n:
            break

        # tokens[i] is now the VALUE (numeric OR allowed literal)
        value_tok = tokens[i]
        i += 1


        # Optional unit immediately after value (only attach if it looks like a unit)
        unit = ""
        if i < n and looks_like_unit(tokens[i]):
            unit = tokens[i]
            i += 1

        parameter = " ".join(param_tokens).strip()
        value_raw = value_tok
        value_num: Optional[float] = None
        if is_number(value_raw):
            try:
                value_num = float(value_raw)
            except Exception:
                value_num = None

        if parameter:
            rows.append(
                {
                    "file": file_name,
                    "page": page_num,
                    "section": current_section or "",
                    "parameter": parameter,
                    "value_raw": value_raw,
                    "value_num": value_num,
                    "unit": unit,
                }
            )
        # Continue; multiple triples may exist per line

    return rows, current_section


def extract_pdf_to_rows(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    all_rows: List[Dict] = []
    current_section: Optional[str] = None

    for p_idx, page in enumerate(reader.pages, start=1):
        # pypdf returns None if no text was found
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            rows, current_section = parse_line_into_rows(
                raw_line, pdf_path.name, p_idx, current_section
            )
            if rows:
                all_rows.extend(rows)

    return all_rows


def collect_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.pdf") if p.is_file()])
    raise FileNotFoundError(f"No PDF found at: {input_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract DCF-style log PDFs into a table (CSV) using pypdf.")
    ap.add_argument("input", help="Input PDF file or a folder containing PDFs (recursive)")
    ap.add_argument("-o", "--output", help="Output CSV path", default="extracted_logs.csv")
    ap.add_argument("--no-section", action="store_true", help="Disable section detection/tagging")
    args = ap.parse_args()

    input_path = Path(args.input)
    pdfs = collect_pdfs(input_path)

    all_rows: List[Dict] = []
    for pdf in pdfs:
        rows = extract_pdf_to_rows(pdf)
        if args.no_section:
            for r in rows:
                r["section"] = ""
        all_rows.extend(rows)

    if not all_rows:
        print("No rows extracted. Check input and consider extending UNITS/SECTION_HEADERS.")
        return

    df = pd.DataFrame(
        all_rows,
        columns=["file", "page", "section", "parameter", "value_raw", "value_num", "unit"],
    )

    # Deduplicate and sort for readability
    df.drop_duplicates(inplace=True)
    #df.sort_values(by=["file", "page", "section", "parameter"], inplace=True)

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()