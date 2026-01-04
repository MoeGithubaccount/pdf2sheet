from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import os
import uuid
import shutil
import traceback
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import pdfplumber

# Camelot is optional at runtime (but installed for you now)
try:
    import camelot  # type: ignore
except Exception:
    camelot = None  # fallback mode


# -------------------------
# App setup
# -------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Helpers: cleaning/scoring
# -------------------------
def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid pandas FutureWarning: applymap deprecated
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda x: str(x).strip() if x is not None else "")
    return df


def _drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize empty strings
    df = df.replace({None: "", "None": ""})

    # drop fully empty rows
    df = df.loc[~(df.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1))]

    # drop fully empty cols
    df = df.loc[:, ~(df.apply(lambda c: all(str(v).strip() == "" for v in c), axis=0))]

    return df


def _basic_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Heuristic score to decide if extraction is "good enough".
    """
    if df is None or df.empty:
        return {"ok": False, "score": 0, "reason": "empty"}

    rows, cols = df.shape
    if rows < 2 or cols < 2:
        return {"ok": False, "score": 0, "reason": f"too small ({rows}x{cols})"}

    # count non-empty cells
    total = rows * cols
    non_empty = 0
    for c in df.columns:
        non_empty += (df[c].astype(str).str.strip() != "").sum()

    density = non_empty / total if total else 0

    # common failure case: a single column of long text or 1-3 cols only
    if cols <= 3 and rows <= 10 and density < 0.35:
        return {"ok": False, "score": 10, "reason": "sparse small table"}

    # score: density + size
    size_bonus = min(40, (rows * cols) // 20)
    score = int(density * 60) + size_bonus

    ok = score >= 35  # threshold
    return {
        "ok": ok,
        "score": score,
        "rows": rows,
        "cols": cols,
        "density": round(density, 3),
        "reason": "ok" if ok else "below threshold",
    }


def _concat_tables_vertical(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Concatenate tables vertically, aligning columns by index where possible.
    If column counts differ, we pad to max columns.
    """
    if not dfs:
        return None

    cleaned = []
    max_cols = max(df.shape[1] for df in dfs if df is not None and not df.empty)

    for df in dfs:
        if df is None or df.empty:
            continue
        df = df.copy()
        # pad columns
        if df.shape[1] < max_cols:
            for _ in range(max_cols - df.shape[1]):
                df[df.shape[1]] = ""
        # rename columns as integers to avoid collisions
        df.columns = list(range(df.shape[1]))
        cleaned.append(df)

    if not cleaned:
        return None

    out = pd.concat(cleaned, ignore_index=True)
    out = _strip_df(out)
    out = _drop_empty_rows_cols(out)
    return out


# -------------------------
# Extractors
# -------------------------
def extract_with_camelot(pdf_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Try Camelot lattice then stream.
    Returns: (df, debug)
    """
    debug: Dict[str, Any] = {"engine": "camelot", "attempts": []}

    if camelot is None:
        debug["error"] = "camelot not installed"
        return None, debug

    # Try lattice first (best with ruled tables)
    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages="all",
                flavor=flavor,
                strip_text="\n",
            )
            dfs = []
            for t in tables:
                # camelot table df is already a DataFrame
                dfs.append(t.df)

            df = _concat_tables_vertical(dfs)
            q = _basic_quality_score(df) if df is not None else {"ok": False, "score": 0}

            debug["attempts"].append(
                {
                    "flavor": flavor,
                    "tables_found": len(tables),
                    "quality": q,
                }
            )

            if df is not None and q.get("ok"):
                debug["selected"] = {"flavor": flavor}
                return df, debug

        except Exception as e:
            debug["attempts"].append(
                {"flavor": flavor, "error": str(e)}
            )

    return None, debug


def extract_tables_pdfplumber(pdf_path: str) -> List[List[List[str]]]:
    """
    Extracts tables as list-of-tables, each table is list-of-rows, each row list-of-cells.
    """
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)
    return all_tables


def tables_to_dataframe(tables: List[List[List[str]]]) -> Optional[pd.DataFrame]:
    """
    Convert pdfplumber tables to a single DataFrame (vertical concat).
    """
    if not tables:
        return None

    dfs = []
    for t in tables:
        if not t:
            continue
        df = pd.DataFrame(t)
        dfs.append(df)

    out = _concat_tables_vertical(dfs)
    return out


def extract_best(pdf_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Best long-term quality strategy:
    1) Camelot lattice -> stream
    2) fallback pdfplumber
    """
    # 1) Camelot
    df, debug = extract_with_camelot(pdf_path)
    if df is not None:
        return df, debug

    # 2) pdfplumber fallback
    debug2: Dict[str, Any] = {"engine": "pdfplumber"}
    try:
        tables = extract_tables_pdfplumber(pdf_path)
        df2 = tables_to_dataframe(tables)
        q = _basic_quality_score(df2) if df2 is not None else {"ok": False, "score": 0}
        debug2["tables_found"] = len(tables)
        debug2["quality"] = q

        if df2 is None or not q.get("ok"):
            # still return df2 if exists, but mark weak
            debug2["warning"] = "Extraction quality low. Consider OCR for scanned PDFs."
        return df2, {"fallback": debug, **debug2}
    except Exception as e:
        debug2["error"] = str(e)
        return None, {"fallback": debug, **debug2}


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    content = await file.read()

    MAX_SIZE = 20 * 1024 * 1024  # 20MB
    if len(content) > MAX_SIZE:
        return {
            "ok": False,
            "error": "File too large. Max size is 20MB."
        }

    with open(pdf_path, "wb") as f:
        f.write(content)

    tables = extract_tables_pdfplumber(pdf_path)
    df = tables_to_dataframe(tables)

    if df is None:
        return {
            "ok": False,
            "error": "No tables detected. Try a clearer PDF."
        }

    csv_path = os.path.join(OUTPUT_DIR, f"{file_id}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"{file_id}.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    return {
        "ok": True,
        "csv": f"/download/{file_id}.csv",
        "xlsx": f"/download/{file_id}.xlsx",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }


@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found."}, status_code=404)
    return FileResponse(path, filename=filename)

