import os
import io
import re
import time
import uuid
import shutil
import logging
import subprocess
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import pdfplumber

# Camelot is optional at runtime; if missing we fall back gracefully
try:
    import camelot  # type: ignore
except Exception:
    camelot = None


APP_NAME = "pdf2sheet"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settings (can be overridden in Render Environment)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "24"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(APP_NAME)

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# (optional) if you have a static folder
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": datetime.utcnow().isoformat()}


def cleanup_old_files() -> None:
    """Delete old uploads/outputs to keep disk from filling up (important on Render)."""
    cutoff = datetime.utcnow() - timedelta(hours=RETENTION_HOURS)
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if os.path.isfile(path):
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                    if mtime < cutoff:
                        os.remove(path)
            except Exception:
                pass


def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "upload.pdf"


def count_pages(pdf_path: str) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def extract_pdfplumber_tables(pdf_path: str) -> List[pd.DataFrame]:
    """Simple table extraction using pdfplumber."""
    dfs: List[pd.DataFrame] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract_table() gets one table; extract_tables() gets multiple
            tables = page.extract_tables()
            for t in tables or []:
                if not t or len(t) < 2:
                    continue
                df = pd.DataFrame(t)
                dfs.append(df)
    return dfs


def extract_camelot_tables(pdf_path: str) -> List[pd.DataFrame]:
    """Stronger table extraction using Camelot (digital PDFs)."""
    if camelot is None:
        return []

    dfs: List[pd.DataFrame] = []
    # Try both flavors; each works better depending on PDF
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
            for t in tables:
                df = t.df
                if df is not None and df.shape[0] >= 2 and df.shape[1] >= 2:
                    dfs.append(df)
        except Exception:
            continue
    return dfs


def normalize_dfs(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Combine multiple extracted tables into one big dataframe."""
    cleaned: List[pd.DataFrame] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        # Trim whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Drop empty rows/cols
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Skip tiny noise
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        cleaned.append(df)

    if not cleaned:
        return None

    # If multiple tables, stack them with a blank row separator
    out = cleaned[0]
    for nxt in cleaned[1:]:
        spacer = pd.DataFrame([[""] * max(out.shape[1], nxt.shape[1])])
        # pad both to same width
        w = max(out.shape[1], nxt.shape[1])
        out = out.reindex(columns=range(w), fill_value="")
        nxt = nxt.reindex(columns=range(w), fill_value="")
        out = pd.concat([out, spacer, nxt], ignore_index=True)

    return out


@app.post("/convert")
async def convert(file: UploadFile = File(...), mode: str = Form("fast")):
    """
    mode:
      - fast: pdfplumber only
      - best: camelot first, then pdfplumber fallback
    """
    cleanup_old_files()

    if file.content_type not in ("application/pdf", "application/x-pdf"):
        return JSONResponse({"ok": False, "error": "Please upload a PDF."}, status_code=400)

    # Size limit protection (Render disk + memory)
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return JSONResponse(
            {"ok": False, "error": f"File too large ({size_mb:.1f}MB). Limit is {MAX_UPLOAD_MB}MB."},
            status_code=413,
        )

    file_id = uuid.uuid4().hex
    original_name = safe_filename(file.filename or "upload.pdf")
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}_{original_name}")

    with open(pdf_path, "wb") as f:
        f.write(content)

    pages = count_pages(pdf_path)
    started = time.time()

    # Extraction pipeline
    dfs: List[pd.DataFrame] = []
    used_mode = mode.lower().strip()

    if used_mode == "best":
        # Try camelot first (best for digital/ruled tables)
        dfs = extract_camelot_tables(pdf_path)

        # Fallback to pdfplumber if camelot found nothing
        if not dfs:
            dfs = extract_pdfplumber_tables(pdf_path)
    else:
        # Fast mode = pdfplumber only
        used_mode = "fast"
        dfs = extract_pdfplumber_tables(pdf_path)

    df = normalize_dfs(dfs)

    if df is None:
        took = time.time() - started
        log.info("No tables detected. mode=%s pages=%s file=%s took=%.2fs", used_mode, pages, original_name, took)
        return {"ok": False, "error": "No tables detected. Try Best mode or a cleaner PDF."}

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, f"{file_id}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"{file_id}.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    took = time.time() - started
    log.info(
        "Converted ok. mode=%s pages=%s tables=%s rows=%s cols=%s file=%s took=%.2fs",
        used_mode, pages, len(dfs), df.shape[0], df.shape[1], original_name, took
    )

    return {
        "ok": True,
        "mode": used_mode,
        "pages": int(pages),
        "tables": int(len(dfs)),
        "csv": f"/download/{file_id}.csv",
        "xlsx": f"/download/{file_id}.xlsx",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }


@app.get("/download/{filename}")
def download(filename: str):
    # basic safety
    filename = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found (expired). Please re-convert."}, status_code=404)
    return FileResponse(path, filename=filename)

