
import os
import re
import time
import uuid
import shutil
import subprocess
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import pdfplumber
import camelot

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool


# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="pdf2sheet", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# repo root = one level above /app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helpers: safety + cleanup
# -----------------------------
def safe_filename(name: str) -> str:
    name = (name or "upload.pdf").strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]+", "", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name or "upload.pdf"


def count_pages(pdf_path: str) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


# -----------------------------
# Helpers: DF normalization
# -----------------------------
def _clean_cell(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() == "nan":
        return ""
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for i, c in enumerate(df.columns):
        c = "" if c is None else str(c)
        c = _clean_cell(c)
        if not c:
            c = f"col_{i+1}"
        cols.append(c)

    # make unique
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
    df = df.applymap(_clean_cell)
    # drop empty columns/rows
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
        cleaned.append(df)

    if not cleaned:
        return pd.DataFrame()

    # stack tables with separator row (keeps multi-table PDFs readable)
    out = []
    for df in cleaned:
        out.append(df)
        out.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))
    merged = pd.concat(out, ignore_index=True)

    # final cleanup
    merged = drop_empty(merged)
    return merged


# -----------------------------
# Helpers: extraction (Fast / Best)
# -----------------------------
def extract_pdfplumber_tables(pdf_path: str, max_pages: Optional[int] = None) -> List[pd.DataFrame]:
    """
    FAST: pdfplumber page.extract_tables().
    """
    dfs: List[pd.DataFrame] = []

    # Table settings help borderline PDFs; these defaults are safe.
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
        "intersection_tolerance": 3,
    }

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page in pages:
            try:
                tables = page.extract_tables(table_settings=table_settings) or []
                for t in tables:
                    if not t or len(t) < 2:
                        continue
                    header = t[0]
                    body = t[1:]
                    df = pd.DataFrame(body, columns=header)
                    dfs.append(df)
            except Exception:
                continue

    return dfs


def extract_camelot_tables(pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
    """
    BEST: try lattice then stream.
    lattice works best when cell borders exist.
    stream works when borders don't exist.
    """
    dfs: List[pd.DataFrame] = []

    # Try lattice
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        if tables and tables.n > 0:
            for t in tables:
                df = t.df
                if df is None or df.empty:
                    continue
                if df.shape[0] >= 2:
                    header = df.iloc[0].tolist()
                    body = df.iloc[1:].reset_index(drop=True)
                    body.columns = header
                    dfs.append(body)
    except Exception:
        pass

    if dfs:
        return dfs

    # Try stream
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")
        if tables and tables.n > 0:
            for t in tables:
                df = t.df
                if df is None or df.empty:
                    continue
                if df.shape[0] >= 2:
                    header = df.iloc[0].tolist()
                    body = df.iloc[1:].reset_index(drop=True)
                    body.columns = header
                    dfs.append(body)
    except Exception:
        pass

    return dfs


# -----------------------------
# Helpers: Auto quality + OCR
# -----------------------------
def has_text_layer(pdf_path: str, max_pages: int = 2) -> bool:
    """If first pages contain selectable text, it's likely NOT scanned."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                txt = (page.extract_text() or "").strip()
                if len(txt) >= 30:
                    return True
        return False
    except Exception:
        return False


def tables_look_weak(dfs: List[pd.DataFrame], min_fill_ratio: float = 0.45) -> bool:
    """
    Treat results as weak if:
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
    return fill_ratio < min_fill_ratio


def ocr_pdf_to_searchable(pdf_path: str, out_path: str) -> bool:
    """
    Convert scanned PDF to searchable PDF using:
    - pdftoppm (poppler-utils) to render images
    - tesseract to OCR each page image into PDF
    - ghostscript to merge PDFs
    """
    workdir = None
    try:
        workdir = os.path.join(UPLOAD_DIR, f"ocr_{uuid.uuid4().hex}")
        os.makedirs(workdir, exist_ok=True)

        # 1) Render pages to PNG
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

        # 2) OCR each image to PDF part
        pdf_parts: List[str] = []
        for img in images:
            base = img[:-4]  # strip .png
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

        # 3) Merge parts into a single searchable PDF
        gs_cmd = [
            "gs",
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={out_path}",
        ] + pdf_parts
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

    if df is None or df.empty:
        empty = pd.DataFrame({"message": ["No tables detected. Try a clearer PDF or OCR."]})
        empty.to_csv(csv_path, index=False)
        empty.to_excel(xlsx_path, index=False)
        return csv_path, xlsx_path

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


# -----------------------------
# HTML UI
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "status": "up"}

@app.head("/")
def head_root():
    return HTMLResponse("")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>pdf2sheet</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family: system-ui; max-width: 860px; margin: 40px auto; padding: 0 12px;">
  <h1 style="margin-bottom: 6px;">pdf2sheet</h1>
  <p style="margin-top: 0; color: #444;">Upload a PDF and download CSV/XLSX.</p>

  <form action="/convert" method="post" enctype="multipart/form-data"
        style="border: 1px solid #eee; padding: 16px; border-radius: 12px;">
    <label style="display:block; font-weight:600;">Mode:</label>
    <select name="mode" style="padding: 8px; margin-top: 8px;">
      <option value="auto" selected>auto (recommended)</option>
      <option value="fast">fast</option>
      <option value="best">best</option>
      <option value="ocr">ocr</option>
    </select>

    <div style="height: 14px;"></div>

    <input type="file" name="file" accept="application/pdf" required style="padding: 8px;" />
    <div style="height: 14px;"></div>

    <button type="submit" style="padding: 10px 14px; border-radius: 10px; border: 0; cursor:pointer;">
      Convert
    </button>

    <p style="margin-top: 12px; color:#666; font-size: 13px;">
      Tip: Use <b>auto</b>. It switches to OCR for scanned PDFs automatically.
    </p>
  </form>
</body>
</html>
"""


# -----------------------------
# Downloads
# -----------------------------
@app.get("/download/{filename}")
def download(filename: str):
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        return JSONResponse({"ok": False, "error": "Invalid file type."}, status_code=400)

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found."}, status_code=404)

    media = "text/csv" if filename.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path, media_type=media, filename=filename)


# -----------------------------
# Convert
# -----------------------------
def _wants_html(request: Request) -> bool:
    """
    If user submits from browser form, they typically accept text/html.
    If API client requests JSON, they send Accept: application/json.
    """
    accept = (request.headers.get("accept") or "").lower()
    # Most browsers send 'text/html,...'
    return "text/html" in accept and "application/json" not in accept


def _html_result(payload: Dict[str, Any]) -> HTMLResponse:
    file_id = payload["file_id"]
    csv_url = payload["downloads"]["csv"]
    xlsx_url = payload["downloads"]["xlsx"]

    return HTMLResponse(
        f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>pdf2sheet — result</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family: system-ui; max-width: 860px; margin: 40px auto; padding: 0 12px;">
  <h1 style="margin-bottom: 6px;">Done ✅</h1>
  <p style="margin-top: 0; color:#444;">
    Mode used: <b>{payload.get("mode_used")}</b> • Pages: <b>{payload.get("pages")}</b> •
    Rows: <b>{payload.get("rows")}</b> • Cols: <b>{payload.get("cols")}</b> •
    Time: <b>{payload.get("elapsed_s")}s</b>
  </p>

  <div style="display:flex; gap:12px; flex-wrap: wrap; margin: 18px 0;">
    <a href="{csv_url}" style="padding: 10px 14px; border-radius: 10px; background:#111; color:#fff; text-decoration:none;">
      Download CSV
    </a>
    <a href="{xlsx_url}" style="padding: 10px 14px; border-radius: 10px; background:#111; color:#fff; text-decoration:none;">
      Download XLSX
    </a>
    <a href="/" style="padding: 10px 14px; border-radius: 10px; border:1px solid #ddd; text-decoration:none; color:#111;">
      Convert another
    </a>
  </div>

  <details style="margin-top: 10px;">
    <summary style="cursor:pointer;">Details</summary>
    <pre style="background:#f6f6f6; padding:12px; border-radius: 10px; overflow:auto;">{payload}</pre>
  </details>
</body>
</html>
"""
    )


@app.post("/convert")
async def convert(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
):
    started = time.time()
    file_id = uuid.uuid4().hex

    original_name = safe_filename(file.filename or "upload.pdf")
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}_{original_name}")

    content = await file.read()
    if not content:
        resp = {"ok": False, "error": "Empty upload."}
        return HTMLResponse(resp["error"], status_code=400) if _wants_html(request) else JSONResponse(resp, status_code=400)

    with open(pdf_path, "wb") as f:
        f.write(content)

    pages = await run_in_threadpool(count_pages, pdf_path)

    used_mode = (mode or "auto").lower().strip()
    if used_mode not in ("auto", "fast", "best", "ocr"):
        used_mode = "auto"

    # Decide scanned vs not
    scanned = await run_in_threadpool(lambda: not has_text_layer(pdf_path))
    working_pdf = pdf_path

    # 1) OCR path (forced or auto+scanned)
    if used_mode == "ocr" or (used_mode == "auto" and scanned):
        ocr_path = os.path.join(UPLOAD_DIR, f"{file_id}_ocr.pdf")
        ok = await run_in_threadpool(ocr_pdf_to_searchable, pdf_path, ocr_path)
        if not ok:
            resp = {"ok": False, "error": "Scanned PDF detected, but OCR failed on the server."}
            return HTMLResponse(resp["error"], status_code=500) if _wants_html(request) else JSONResponse(resp, status_code=500)

        working_pdf = ocr_path
        used_mode = "ocr"

        # OCR behaves like Best afterward
        dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
        if not dfs:
            dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

    # 2) Best forced
    elif used_mode == "best":
        dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
        if not dfs:
            dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

    # 3) Fast or Auto (non-scanned)
    else:
        # Fast first
        dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

        # Auto upgrade if empty or weak
        if used_mode == "auto" and (not dfs or await run_in_threadpool(tables_look_weak, dfs)):
            used_mode = "best"
            dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
            if not dfs:
                dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)
        else:
            used_mode = "fast"

        # Optional fallback even if user forced fast: if nothing found, try best
        if used_mode == "fast" and not dfs:
            used_mode = "best"
            dfs = await run_in_threadpool(extract_camelot_tables, working_pdf)
            if not dfs:
                dfs = await run_in_threadpool(extract_pdfplumber_tables, working_pdf)

    df = await run_in_threadpool(normalize_dfs, dfs)
    await run_in_threadpool(write_outputs, df, file_id)

    elapsed = round(time.time() - started, 3)

    payload = {
        "ok": True,
        "file_id": file_id,
        "mode_used": used_mode,
        "pages": pages,
        "rows": int(df.shape[0]) if df is not None and not df.empty else 0,
        "cols": int(df.shape[1]) if df is not None and not df.empty else 0,
        "elapsed_s": elapsed,
        "decision": {
            "requested_mode": (mode or "auto").lower().strip(),
            "scanned_detected": bool(scanned),
        },
        "downloads": {
            "csv": f"/download/{file_id}.csv",
            "xlsx": f"/download/{file_id}.xlsx",
        },
    }

    if _wants_html(request):
        return _html_result(payload)

    return JSONResponse(payload)


# -----------------------------
# Debug: confirm binaries in Render container
# -----------------------------
@app.get("/debug/bins")
def debug_bins():
    import shutil as _shutil
    return {
        "tesseract": _shutil.which("tesseract"),
        "pdftoppm": _shutil.which("pdftoppm"),
        "gs": _shutil.which("gs"),
    }


