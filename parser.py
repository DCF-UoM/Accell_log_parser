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
from datetime import datetime

import pandas as pd
from pypdf import PdfReader



# --- Configuration you can tweak ------------------------------------------------

# Section headers seen in these logs; extend if you encounter new ones.
SECTION_HEADERS = [
    # Sources
    "Source 2",
    "Source 1",
    "Source B",

    # Accelerator blocks
    "Pre Acceleration",
    "Accelerator",
    "Post Acceleration",

    # Facility/line-specific blocks
    "7.5SH-2 Accelerator",
    "7.5SH-2 Vacuum",
    "7.5SH-2 Machine Setup",

    # Beamlines (observed)
    "+15 Degree Beamline",
    "-15 Degree Beamline",
    "-4 Degree Beamline",
    "B1 Beamline",

    # Optional/seen in earlier configs (keep for future logs)
    "+30 Degree Beamline",
    "-30 Degree Beamline",
    "A1 Beamline",

    # Global
    "Faraday Cups",
    "Vacuum",
    "Machine Setup",
]

# Common units found in the logs; extend as needed (e.g., add 'mbar', 'Hz', 's', etc.).
UNITS = {
    # Flows & currents & voltages
    "sccm", "A", "mA", "uA", "V", "kV",

    # Fields, pressure, position
    "G", "T", "mm", "psig", "Pa/Ta",

    # Temperatures
    "DegC", "C",

    # Energies and scales
    "MV", "MeV", "%", "AMU", "m/q",

    # Misc (observed in logs)
    "W", "kW", "u", "Trn",
}

# Optional: helps skip non-data prefixes faster; not strictly required.
LIKELY_PARAM_STARTS = {
    # Common families already in your script
    "MV","FIL","ARC","EXT","FOC","GAP","BIA","OVN","CHM","INJ","IGC","ES","EL",
    "IM","TNK","CH","COL","CPS","TPS","GS","MQ","MS","SM","HPB","SS","IP","CVG","FC",
    "SETUP", "PRB", "MAG", "VEL", "RAS", "DS", "ION", "IML", "CAT", "MCS", "SETUP-B",
}

SUFFIX_TAGS = {
    # Short, universal tags
    "VC","VR","CR","CC","PR","DR","DC","TR","WR",

    # Axis/corrector tags
    "XVC","YVC","XVR","+XVR","-XVR","+YVR","-YVR","+VR","-VR",

    # TPS/energy/control family
    "GvmVR","TrvVC","ModeSR","LEsltCR","HEsltCR","CtlGain","CPOgain",

    # Probe and grid
    "PrbDC","PrbDR","PrbCR","PrbQCC","GridVR",

    # Magnets / misc readbacks
    "Strength","Balance","XCR","YCR","YCC","XCC","MfieldR","LastCR","CatNumR",

    # Injection/calcs
    "TotInjV","VELcalc","m/q_calc","k",

    # States & selectors
    "PosSR","PosSC","SyncSC",

    # Machine setup
    "TotPartE","TotMachE","TotInjE","Ispecies","Ospecies","ChgState","AtomNumb","SrcSel","BLsel",

    # Site-specific oddities
    "CRlost",   # e.g., CH TX-n CRlost
}

# Literal value
VALUE_LITERALS = {
    # States / modes
    "open", "closed", "moving", "internal", "gvm", "SNICS", "TORVIS",

    # Element symbols observed
    "H", "He", "Kr", "Au",

    # Selector codes commonly used as values
    "S1", "S2",
    "LA", "LB", "L3", "L5",
    # (Feel free to extend with others as they appear)
}

# Any token ending with SR or Sel is already handled in your code (…SR / …Sel).
# These are extra, explicit keys observed to take literal values:
LITERAL_VALUE_KEYS = {
    "AtomNumb",  # H, He, Kr, Au, ...
    "SrcSel",    # S1, S2, ...
    "BLsel",     # LA, L3, L5, ...
    "PosSC",     # open/closed
    "SyncSC",    # internal
    # SR keys (redundant but harmless if you want to be explicit)
    "PolSR", "ModeSR", "PosSR",
}
INDEX_TOKEN_RE = re.compile(r"^[+-]?\d+(?:-\d+)?$")  # -1, 1, 01-1, 2-3, etc.

# --- Helpers --------------------------------------------------------------------

# --- Section header detection (with optional prefix code) -----------------------

# Prefer longer headers first (e.g., "Source 2" before "Source")
_HEADERS_SORTED = sorted(SECTION_HEADERS, key=len, reverse=True)
_HEADER_ALT = "|".join(map(re.escape, _HEADERS_SORTED))

# A "section code" immediately before the header, e.g., "7.5SH-2 Machine Setup"
#   - starts with an alphanumeric
#   - may contain letters, digits, dots, or hyphens
# We require the match to start at a word boundary (or line start) and end at a word boundary.
SECTION_RE = re.compile(
    rf"""
    (?<!\S)                                   # start of line or whitespace (no non-space before)
    (?:
        (?P<prefix>[A-Za-z0-9][A-Za-z0-9.\-]*)\s+   # optional code like 7.5SH-2
    )?
    (?P<header>(?:{_HEADER_ALT}))             # one of the known headers
    (?=\s|$)                                  # followed by space or end of line
    """,
    re.IGNORECASE | re.VERBOSE,
)

NUM_RE = re.compile(
    r"""
    ^[+-]?(
        (?:\d+\.\d*|\.\d+|\d+)       # int or float
        (?:[eE][+-]?\d+)?            # optional exponent
    )$
    """,
    re.VERBOSE,
)

# Timestamp at start of the first line, e.g. "19-Dec-2025 10:33:01 DCF Dual Beam ..."
DATE_RE = re.compile(r"(?m)^\s*(\d{2}-[A-Za-z]{3}-\d{4})\s+(\d{2}:\d{2}:\d{2})\b")

def extract_page_timestamp(page_text: str) -> Optional[str]:
    """
    Find the 'DD-Mon-YYYY HH:MM:SS' timestamp in a page and return ISO 'YYYY-MM-DD HH:MM:SS'.
    Returns None if not found or parse fails.
    """
    m = DATE_RE.search(page_text)
    if not m:
        return None
    dt_raw = f"{m.group(1)} {m.group(2)}"
    try:
        # %b = locale-independent English month abbrev in Python's datetime
        dt = datetime.strptime(dt_raw, "%d-%b-%Y %H:%M:%S")
        return dt.isoformat(sep=" ", timespec="seconds")
    except Exception:
        # Fall back to raw if strict parsing fails (rare)
        return dt_raw

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
      - the preceding param token expects a literal (…SR / …Sel / explicit list),
        and this token is not a number (uppercase codes like LA, S2, L3 are allowed).
    """
    if tok in VALUE_LITERALS:
        return True
    if param_tokens and expects_literal_after(param_tokens[-1]):
        # IMPORTANT: do not exclude tokens that look like units or parameter starts;
        # in this context, codes like LA, S2, L3 are valid literal values.
        return not is_number(tok)
    return False

def is_index_token(tok: str) -> bool:
    # Numeric tokens that are often part of a parameter path, not the value
    return bool(INDEX_TOKEN_RE.match(tok))

def is_suffix_tag(tok: str) -> bool:
    """Only treat it as a suffix tag if it is explicitly listed."""
    if tok in SUFFIX_TAGS:
        return True
    # Handle +VR / -VR style by stripping the sign and checking again
    if tok and tok[0] in {"+", "-"} and tok[1:] in SUFFIX_TAGS:
        return True
    return False

def find_section_in_line(line: str) -> Tuple[Optional[str], str]:
    """
    Detect a section header in the line, optionally preceded by a code like '7.5SH-2'.
    Returns (full_section_text, line_with_that_span_removed) or (None, original_line).
    """
    m = SECTION_RE.search(line)
    if not m:
        return None, line

    prefix = m.group("prefix") or ""
    header = m.group("header")
    full = (prefix + " " + header).strip()

    # Remove the entire matched slice so prefix/header tokens don't leak into parsing
    new_line = (line[:m.start()] + " " + line[m.end():]).strip()
    return full, new_line



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

    
    # Strip document-banner text if present, but keep the rest of the line
    work_line = re.sub(
        r"""\bDCF\s+Dual\s+Beam\b(?:\s+Page\s+\d+\s+of\s+\d+)?""",
        "",
        work_line,
        flags=re.IGNORECASE,
    ).strip()

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
        page_ts = extract_page_timestamp(page_text)
        for raw_line in page_text.splitlines():
            rows, current_section = parse_line_into_rows(
                raw_line, pdf_path.name, p_idx, current_section
            )
            if rows:
                for r in rows:
                    r["timestamp"] = page_ts or ""

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
        columns=["file", "page", "timestamp", "section", "parameter", "value_raw", "value_num", "unit"],
    )

    # Deduplicate and sort for readability
    df.drop_duplicates(inplace=True)
    #df.sort_values(by=["file", "page", "section", "parameter"], inplace=True)

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()