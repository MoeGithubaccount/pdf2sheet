import os
import re
import time
import uuid
import shutil
import subprocess
from typing import List, Optional, Tuple

import pandas as pd
import pdfplumber
import camelot

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool


# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="pdf2sheet", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (../)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helpers: file + text cleanup
# -----------------------------
def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]+", "", name)
    return name or "upload.pdf"


def count_pages(pdf_path: str) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make column names reasonable + unique
    cols = []
    for i, c in enumerate(df.columns):
        c = "" if c is None else str(c)
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            c = f"col_{i+1}"
        cols.append(c)

    # ensure unique
    seen = {}
    uniq = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}_{seen[c]}")
    df.columns = uniq
    return df


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trim whitespace in all cells
    df = df.applymap(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else "")
    # Drop empty rows/cols
    df = df.loc[:, (df != "").any(axis=0)]
    df = df.loc[(df != "").any(axis=1), :]
    return df


def normalize_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    cleaned: List[pd.DataFrame] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        df = drop_empty(df)
        if df.empty:
            continue
        df = normalize_columns(df)

        # Sometimes Camelot creates unnamed header rows; keep as-is but ensure not empty
        cleaned.append(df)

    if not cleaned:
        return pd.DataFrame()

    # If multiple tables, stack them with a blank row between tables for readability
    out = []
    for df in cleaned:
        out.append(df)
        out.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))
    merged = pd.concat(out, ignore_index=True)
    merged = drop_empty(merged) if not merged.empty else merged
    return merged


# -----------------------------
# Helpers: extraction
# -----------------------------
def extract_pdfplumber_tables(pdf_path: str, max_pages: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Fast mode: pdfplumber table extraction. Best for digital PDFs with clean lines/text.
    """
    dfs: List[pd.DataFrame] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page in pages:
            try:
                tables = page.extract_tables() or []
                for t in tables:
                    if not t or len(t) < 2:
                        continue
                    df = pd.DataFrame(t[1:], columns=t[0])
                    dfs.append(df)
            except Exception:
                continue
    return dfs


def extract_camelot_tables(pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
    """
    Best mode: Camelot (try lattice then stream). Falls back handled outside.
    """
    dfs: List[pd.DataFrame] = []

    # Camelot can be sensitive. We try lattice first, then stream.
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
            if tables and tables.n > 0:
                for t in tables:
                    df = t.df
                    if df is None or df.empty:
                        continue
                    # Camelot returns all strings; treat first row as header if it looks like one
                    # We'll keep it simple: use first row as header, rest as rows
                    if df.shape[0] >= 2:
                        header = df.iloc[0].tolist()
                        body = df.iloc[1:].reset_index(drop=True)
                        body.columns = header
                        dfs.append(body)
            if dfs:
                return dfs
        except Exception:
            continue

    return dfs


# -----------------------------
# Helpers: "smart" auto quality
# -----------------------------
def has_text_layer(pdf_path: str, max_pages: int = 2) -> bool:
    """Quick check: if first pages contain selectable text, it’s likely NOT scanned."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                txt = (page.extract_text() or "").strip()
                if len(txt) >= 30:
                    return True
        return False
    except Exception:
        return False


def tables_look_weak(dfs: List[pd.DataFrame]) -> bool:
    """
    Heuristic: treat results as weak if:
    - no dfs
    - only tiny tables
    - mostly empty cells
    """
    if not dfs:
        return True

    total_cells = 0
    non_empty = 0
    meaningful_tables = 0

    for df in dfs:
        if df is None or df.empty:
            continue
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        meaningful_tables += 1
        vals = df.astype(str).replace("nan", "").values
        total_cells += vals.size
        non_empty += (vals != "").sum()

    if meaningful_tables == 0 or total_cells == 0:
        return True

    fill_ratio = non_empty / total_cells
    return fill_ratio < 0.55


def ocr_pdf_to_searchable(pdf_path: str, out_path: str) -> bool:
    """
    Convert scanned PDF to searchable PDF using Tesseract.
    Requires: tesseract-ocr + poppler-utils (pdftoppm) + ghostscript in Docker.
    """
    workdir = None
    try:
        workdir = os.path.join(UPLOAD_DIR, f"ocr_{uuid.uuid4().hex}")
        os.makedirs(workdir, exist_ok=True)

        # PDF -> PNG pages
        prefix = os.path.join(workdir, "page")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "250", pdf_path, prefix],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        images = sorted(
            os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(".png")
        )
        if not images:
            return False

        # OCR each image -> PDF part
        pdf_parts = []
        for img in images:
            base = img[:-4]  # remove .png
            subprocess.run(
                ["tesseract", img, base, "pdf"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            part_pdf = base + ".pdf"
            if os.path.exists(part_pdf):
                pdf_parts.append(part_pdf)

        if not pdf_parts:
            return False

        # Merge using ghostscript
        gs_cmd = ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", f"-sOutputFile={out_path}"] + pdf_parts
        subprocess.run(gs_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return True
    except Exception:
        return False
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


# -----------------------------
# Output writers
# -----------------------------
def write_outputs(df: pd.DataFrame, file_id: str) -> Tuple[str, str]:
    csv_path = os.path.join(OUTPUT_DIR, f"{file_id}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"{file_id}.xlsx")

    # Always write something (even empty) so UI doesn't break
    if df is None or df.empty:
        empty = pd.DataFrame({"message": ["No tables were detected. Try Best/OCR or a clearer PDF."]})
        empty.to_csv(csv_path, index=False)
        empty.to_excel(xlsx_path, index=False)
        return csv_path, xlsx_path

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


# -----------------------------
# UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    # Simple built-in UI (works even if templates/ isn't mounted)
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>pdf2sheet</title></head>
<body style="font-family: system-ui; max-width: 760px; margin: 40px auto;">
  <h1>pdf2sheet</h1>
  <p>Upload a PDF and download CSV/XLSX.</p>
  <form action="/convert" method="post" enctype="multipart/form-data">
    <label>Mode:</label>
    <select name="mode">
      <option value="auto" selected>auto (recommended)</option>
      <option value="fast">fast</option>
      <option value="best">best</option>
      <option value="ocr">ocr</option>
    </select>
    <br><br>
    <input type="file" name="file" accept="application/pdf" required />
    <br><br>
    <button type="submit">Convert</button>
  </form>
  <hr>
  <p><small>Tip: Use <b>auto</b>. It switches to OCR for scanned PDFs automatically.</small></p>
</body>
</html>
"""


# -----------------------------
# Download endpoints
# -----------------------------
@app.get("/download/{filename}")
def download(filename: str):
    # Only allow .csv or .xlsx from output dir
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        return JSONResponse({"ok": False, "error": "Invalid file type."}, status_code=400)

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found."}, status_code=404)

    media = "text/csv" if filename.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path, media_type=media, filename=filename)


# -----------------------------
# Convert endpoint
# -----------------------------
@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
):
    started = time.time()
    file_id = uuid.uuid4().hex

    # Save upload
    original_name = safe_filename(file.filename or "upload.pdf")
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}_{original_name}")

    content = await file.read()
    if not content:
        return JSONResponse({"ok": False, "error": "Empty upload."}, status_code=400)

    with open(pdf_path, "wb") as f:
        f.write(content)

    pages = count_pages(pdf_path)

    # Normalize mode
    used_mode = (mode or "auto").lower().strip()
    if used_mode not in ("auto", "fast", "best", "ocr"):
        used_mode = "auto"

    # AUTO QUALITY LOGIC:
    # 1) If scanned (no text layer) -> OCR -> treat as "best"
    # 2) Otherwise: try Fast first, if no tables OR weak tables -> Best
    working_pdf = pdf_path
    scanned = await run_in_threadpool(lambda: not has_text_layer(pdf_path))

    if used_mode in ("ocr",) or (used_mode == "auto" and scanned):
        used_mode = "ocr"
        ocr_path = os.path.join(UPLOAD_DIR, f"{file_id}_ocr.pdf")
        ok = await run_in_threadpool(ocr_pdf_to_searchable, pdf_path, ocr_path)
        if not ok:
            return JSONResponse(
                {"ok": False, "error": "Scanned PDF detected, but OCR failed on the server."},
                status_code=500,
            )
        working_pdf = ocr_path

        # OCR acts like Best extraction afterwards
        dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
        if not dfs:
            dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)
        used_mode = "ocr"

    elif used_mode == "best":
        dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
        if not dfs:
            dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

    else:
        # fast or auto (non-scanned)
        dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

        # If auto: upgrade to best when weak or empty
        if used_mode == "auto" and (not dfs or await run_in_threadpool(tables_look_weak, dfs)):
            used_mode = "best"
            dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
            if not dfs:
                dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)
        else:
            used_mode = "fast"

        # If user forced fast: still allow fallback if nothing found (optional)
        if used_mode == "fast" and not dfs:
            used_mode = "best"
            dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
            if not dfs:
                dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

    df = await run_in_threadpool(normalize_dfs, dfs)
    csv_path, xlsx_path = await run_in_threadpool(write_outputs, df, file_id)

    elapsed = round(time.time() - started, 3)
    return {
        "ok": True,
        "file_id": file_id,
        "mode_used": used_mode,
        "pages": pages,
        "rows": 0 if df is None else int(df.shape[0]),
        "cols": 0 if df is None else int(df.shape[1]),
        "elapsed_s": elapsed,
        "downloads": {
            "csv": f"/download/{file_id}.csv",
            "xlsx": f"/download/{file_id}.xlsx",
        },
    }


# Optional: sanity endpoint to verify binaries exist inside Render container
@app.get("/debug/bins")
def debug_bins():
    import shutil as _shutil
    return {
        "tesseract": _shutil.which("tesseract"),
        "pdftoppm": _shutil.which("pdftoppm"),
        "gs": _shutil.which("gs"),
    }

