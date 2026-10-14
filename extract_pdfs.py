"""
Step 1 of the pipeline: ICMR PDF  ->  clean .txt (one "--- Page N ---" block per page).

What changed compared with the first version
--------------------------------------------
1. use_text_flow=True : ICMR STWs are infographics (boxes, columns, flowcharts).
   Default pdfplumber extraction sorts words by position (top->bottom, left->right),
   so text from boxes that sit side by side gets INTERLEAVED line by line.
   use_text_flow=True reads text in the order the PDF author placed it, which keeps
   every box together (e.g. the whole 'MANAGEMENT / PHC level' box stays in one piece).
   A safety fallback to the default method is kept in case a PDF has no usable flow.
2. Noise cleaning : rotated watermark letters printed one-per-line, the ICMR
   disclaimer/copyright block, repeated ministry header lines and the reference list.
"""

import os
import re
import shutil

import pdfplumber

MIN_PAGE_CHARS = 200  # pages shorter than this are probably images/blank -> warn


def drop_references(text: str) -> str:
    """Removes the numbered reference list (keeps everything after it that is not a reference)."""
    out, skipping = [], False
    for ln in text.split("\n"):
        if re.fullmatch(r"REFERENCES?:?", ln.strip(), flags=re.IGNORECASE):
            skipping = True
            continue
        if skipping:
            is_abbreviation = re.match(r"^[A-Z]{2,}:\s", ln) is not None   # e.g. "CT: Computerized tomogram"
            is_heading = ln.isupper() and len(ln) > 3                       # e.g. "ABBREVIATIONS"
            if is_abbreviation or is_heading:
                skipping = False
            else:
                continue
        out.append(ln)
    return "\n".join(out)


def clean_page_text(text: str) -> str:
    """Generic noise removal for one page of an ICMR STW."""
    text = re.sub(r"[ \t]{2,}", " ", text)

    # rotated watermarks / dates come out as ONE CHARACTER PER LINE ("s\nt\nw\n.\ni\nc\nm\nr...")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) > 1]
    text = "\n".join(lines)

    # standard ICMR disclaimer + copyright block (same on every sheet)
    text = re.sub(r"This STW has been prepared.*?©[^\n]*", "", text, flags=re.DOTALL)

    # repeated ministry header lines and any horizontal watermark line
    boilerplate = {
        "Department of Health Research",
        "Ministry of Health and Family Welfare, Government of India",
    }
    lines = [
        ln for ln in text.split("\n")
        if ln.strip() and ln.strip() not in boilerplate and "stw.icmr" not in ln.lower()
    ]

    text = drop_references("\n".join(lines))
    return re.sub(r"\n{2,}", "\n", text).strip()


def extract_page_text(page) -> str:
    flow = page.extract_text(use_text_flow=True) or ""
    std = page.extract_text() or ""
    chosen = flow if len(flow) >= 0.8 * len(std) else std   # safety fallback
    return clean_page_text(chosen)


def extract_all_pdfs(input_folder="medical_guidelines", output_folder="medical_guidelines_text"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"[*] Wiped old '{output_folder}' directory clean.")
    os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        print(f"[!] The folder '{input_folder}' does not exist.")
        return

    pdf_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".pdf"))
    print(f"[*] Found {len(pdf_files)} PDFs. Starting extraction...\n")

    suspicious = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_name)
        txt_path = os.path.join(output_folder, os.path.splitext(pdf_name)[0] + ".txt")

        try:
            blocks, n_pages = [], 0
            with pdfplumber.open(pdf_path) as pdf:
                n_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = extract_page_text(page)
                    if len(text) < MIN_PAGE_CHARS:
                        suspicious.append((pdf_name, page_num, len(text)))
                        if not text:
                            continue
                    blocks.append(f"--- Page {page_num} ---\n{text}")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(blocks).strip())

            print(f"[+] {pdf_name}: {n_pages} page(s) -> {len(blocks)} saved")

        except Exception as e:
            print(f"[!] Error processing {pdf_name}: {e}")

    if suspicious:
        print("\n[!] Pages with very little text (image-only / blank? may need OCR):")
        for name, pg, n in suspicious:
            print(f"    - {name}  page {pg}  ({n} chars)")

    print(f"\n[*] Done. Clean text files are in '{output_folder}'.")


if __name__ == "__main__":
    extract_all_pdfs()