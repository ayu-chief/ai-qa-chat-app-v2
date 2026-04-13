import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gspread
import pandas as pd
import streamlit as st
from dateutil import parser as date_parser
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer, util


# =========================
# 基本設定
# =========================
st.set_page_config(page_title="白井先生QA集リコメンドチャット v2", layout="centered")
st.title("白井先生QA集リコメンドチャット v2")
st.caption("Google Sheets + Google Docs 両対応 / 最終更新から1時間後に反映 / 意味検索対応")

SPREADSHEET_ID = "1hqwp1bvipMTPMiucddGz7_DtkE-dC9TAK4-BoD3yhrM"
WORKSHEET_GID = 960415359
DOC_ID = "1cDCIBNhPV37HeFT3Wtl6Dt18iz74mee5NsXbCJeuV5E"

DELAY_MINUTES = 60
CACHE_TTL_SECONDS = 300
JST = timezone(timedelta(hours=9))

# 意味検索モデル
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# 認証
# =========================
def get_google_creds() -> Credentials:
    info = st.secrets["google_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/documents.readonly",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ]
    return Credentials.from_service_account_info(info, scopes=scopes)


def get_gspread_client():
    creds = get_google_creds()
    return gspread.authorize(creds)


def get_drive_service():
    creds = get_google_creds()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_docs_service():
    creds = get_google_creds()
    return build("docs", "v1", credentials=creds, cache_discovery=False)


# =========================
# 共通ユーティリティ
# =========================
def parse_google_timestamp(ts: str) -> datetime:
    dt = date_parser.isoparse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_jst(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    return dt.astimezone(JST).strftime("%Y-%m-%d %H:%M")


def get_drive_file_metadata(file_id: str) -> Dict[str, Any]:
    drive = get_drive_service()
    return (
        drive.files()
        .get(fileId=file_id, fields="id,name,mimeType,modifiedTime")
        .execute()
    )


def is_eligible_by_delay(modified_time_str: str, delay_minutes: int = DELAY_MINUTES) -> bool:
    modified_dt = parse_google_timestamp(modified_time_str)
    now = datetime.now(timezone.utc)
    return now >= (modified_dt + timedelta(minutes=delay_minutes))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\u3000", " ").strip()


def is_date_like(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$", text))


def clean_sheet_value(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for r in records:
        key = (
            r.get("source_type", ""),
            r.get("source_date", ""),
            r.get("campus", ""),
            r.get("question", ""),
            r.get("answer", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# =========================
# Sheets 読み込み
# =========================
def find_worksheet_by_gid(spreadsheet, gid: int):
    for ws in spreadsheet.worksheets():
        if getattr(ws, "id", None) == gid:
            return ws
    raise ValueError(f"gid={gid} のワークシートが見つかりません。")


def parse_sheet_rows(values: List[List[Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    current_date = ""
    current_campus = ""
    pending_q = ""

    campus_candidates = {
        "京都", "名古屋", "木更津", "川崎", "尼崎", "盛岡", "湘南台", "町田", "大森", "その他"
    }

    for row in values:
        row = list(row) + ["", "", ""]
        col0 = clean_sheet_value(row[0])
        col1 = clean_sheet_value(row[1])
        col2 = normalize_text(row[2]).strip()

        if is_date_like(col0):
            current_date = col0[:10]
            current_campus = ""
            pending_q = ""
            continue

        if col0 in campus_candidates:
            current_campus = col0
            pending_q = ""
            continue

        if col1 == "Q":
            pending_q = normalize_text(col2)
            continue

        if col1 == "A":
            answer = normalize_text(col2)
            question = normalize_text(pending_q)

            if current_date and current_campus and question and answer:
                records.append(
                    {
                        "source_type": "sheet",
                        "source_date": current_date,
                        "campus": current_campus,
                        "category": "",
                        "question": question,
                        "answer": answer,
                    }
                )
            pending_q = ""

    return dedupe_records(records)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_sheet_records() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = get_drive_file_metadata(SPREADSHEET_ID)

    if not is_eligible_by_delay(meta["modifiedTime"], DELAY_MINUTES):
        return [], {
            "name": meta.get("name", "Sheets"),
            "modifiedTime": meta["modifiedTime"],
            "eligible": False,
            "reason": f"最終更新から {DELAY_MINUTES} 分未満のため保留",
        }

    gc = get_gspread_client()
    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = find_worksheet_by_gid(sh, WORKSHEET_GID)
    values = ws.get_all_values()
    records = parse_sheet_rows(values)

    return records, {
        "name": meta.get("name", "Sheets"),
        "modifiedTime": meta["modifiedTime"],
        "eligible": True,
        "reason": "",
    }


# =========================
# Docs 読み込み
# =========================
def read_structural_elements(elements: List[Dict[str, Any]], lines: List[str]) -> None:
    for value in elements:
        if "paragraph" in value:
            elements_list = value.get("paragraph", {}).get("elements", [])
            text_runs = []
            for elem in elements_list:
                text_run = elem.get("textRun", {})
                content = text_run.get("content", "")
                if content:
                    text_runs.append(content)
            para_text = "".join(text_runs).replace("\n", "").strip()
            if para_text:
                lines.append(para_text)

        elif "table" in value:
            table = value.get("table", {})
            for row in table.get("tableRows", []):
                for cell in row.get("tableCells", []):
                    read_structural_elements(cell.get("content", []), lines)

        elif "tableOfContents" in value:
            toc = value.get("tableOfContents", {})
            read_structural_elements(toc.get("content", []), lines)


def extract_doc_lines(doc: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    tabs = doc.get("tabs", [])
    if tabs:
        for tab in tabs:
            doc_tab = tab.get("documentTab", {})
            content = doc_tab.get("body", {}).get("content", [])
            read_structural_elements(content, lines)
    else:
        content = doc.get("body", {}).get("content", [])
        read_structural_elements(content, lines)

    return [normalize_text(x) for x in lines if normalize_text(x)]


def parse_doc_lines(lines: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    doc_date = ""
    current_campus = ""
    current_question = ""
    current_category = ""
    answer_parts: List[str] = []

    def flush():
        nonlocal current_question, current_category, answer_parts
        q = normalize_text(current_question)
        a = normalize_text(" ".join(answer_parts))
        if current_campus and q and a:
            records.append(
                {
                    "source_type": "doc",
                    "source_date": doc_date,
                    "campus": current_campus,
                    "category": current_category,
                    "question": q,
                    "answer": a,
                }
            )
        current_question = ""
        current_category = ""
        answer_parts = []

    for line in lines[:10]:
        m = re.match(r"^(\d{4}/\d{2}/\d{2})$", line)
        if m:
            doc_date = m.group(1).replace("/", "-")
            break

    for line in lines:
        line = normalize_text(line)

        campus_match = re.match(r"^■(.+?)(校)?$", line)
        if campus_match:
            flush()
            current_campus = campus_match.group(1).strip()
            continue

        q_match = re.match(r"^\*?\s*質問(?:①|②|\d+)?[:：]\s*(.+)$", line)
        if q_match:
            flush()
            current_question = q_match.group(1).strip()
            continue

        c_match = re.match(r"^\*?\s*カテゴリ[:：]\s*(.+)$", line)
        if c_match:
            current_category = c_match.group(1).strip()
            continue

        a_match = re.match(r"^\*?\s*回答の要点[:：]\s*(.+)$", line)
        if a_match:
            answer_parts = [a_match.group(1).strip()]
            continue

        if current_question:
            if line.startswith("*") or line.startswith("・") or line.startswith("-"):
                answer_parts.append(line.lstrip("*・- ").strip())
                continue

    flush()
    return dedupe_records(records)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_doc_records() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = get_drive_file_metadata(DOC_ID)

    if not is_eligible_by_delay(meta["modifiedTime"], DELAY_MINUTES):
        return [], {
            "name": meta.get("name", "Docs"),
            "modifiedTime": meta["modifiedTime"],
            "eligible": False,
            "reason": f"最終更新から {DELAY_MINUTES} 分未満のため保留",
        }

    docs = get_docs_service()
    doc = docs.documents().get(documentId=DOC_ID, includeTabsContent=True).execute()
    lines = extract_doc_lines(doc)
    records = parse_doc_lines(lines)

    return records, {
        "name": meta.get("name", "Docs"),
        "modifiedTime": meta["modifiedTime"],
        "eligible": True,
        "reason": "",
    }


# =========================
# 全件ロード
# =========================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_all_qa() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sheet_records, sheet_meta = load_sheet_records()
    doc_records, doc_meta = load_doc_records()

    all_records = sheet_records + doc_records
    df = pd.DataFrame(all_records)

    if df.empty:
        df = pd.DataFrame(columns=[
            "source_type", "source_date", "campus", "category", "question", "answer"
        ])
    else:
        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)

    meta = {
        "sheet": sheet_meta,
        "doc": doc_meta,
        "count": len(df),
    }
    return df, meta


# =========================
# 意味検索
# =========================
@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def build_search_corpus(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    return (
        df["question"].fillna("") + " " +
        df["answer"].fillna("") + " " +
        df["campus"].fillna("") + " " +
        df["category"].fillna("")
    ).tolist()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def build_corpus_embeddings(texts: List[str]):
    if not texts:
        return None
    model = load_embedding_model()
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def search_qa(df: pd.DataFrame, user_input: str, top_n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df

    texts = build_search_corpus(df)
    corpus_embeddings = build_corpus_embeddings(texts)
    if corpus_embeddings is None:
        return df.iloc[0:0].copy()

    model = load_embedding_model()
    query_embedding = model.encode(
        [user_input],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_n)[0]
    hit_indices = [hit["corpus_id"] for hit in hits]
    hit_scores = [float(hit["score"]) for hit in hits]

    result = df.iloc[hit_indices].copy().reset_index(drop=True)
    result["score"] = hit_scores
    return result


# =========================
# UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("最新データに更新"):
        st.cache_data.clear()
        st.rerun()

with st.expander("データ反映状況"):
    try:
        _, meta_preview = load_all_qa()
        st.write(f"件数: {meta_preview['count']} 件")

        st.markdown("**Sheets**")
        st.write(f"ファイル名: {meta_preview['sheet']['name']}")
        st.write(f"最終更新: {format_jst(parse_google_timestamp(meta_preview['sheet']['modifiedTime']))}")
        st.write(f"取込対象: {'はい' if meta_preview['sheet']['eligible'] else 'まだ'}")
        if meta_preview["sheet"]["reason"]:
            st.caption(meta_preview["sheet"]["reason"])

        st.markdown("**Docs**")
        st.write(f"ファイル名: {meta_preview['doc']['name']}")
        st.write(f"最終更新: {format_jst(parse_google_timestamp(meta_preview['doc']['modifiedTime']))}")
        st.write(f"取込対象: {'はい' if meta_preview['doc']['eligible'] else 'まだ'}")
        if meta_preview["doc"]["reason"]:
            st.caption(meta_preview["doc"]["reason"])

    except Exception as e:
        st.error(f"データ状況の取得に失敗しました: {e}")

with st.form(key="chat_form", clear_on_submit=False):
    user_input = st.text_input("知りたいこと・悩みを入力してください", key="user_input")
    st.markdown("""
##### 入力例
- 例1：「不登校」
- 例2：「友人とのトラブルがあったときの対応は？」
- 例3：「保護者対応で境界線をどう引く？」
    """)
    search_btn = st.form_submit_button("検索")

if search_btn and user_input:
    try:
        with st.spinner("検索中..."):
            time.sleep(0.2)
            df, meta = load_all_qa()
            result_df = search_qa(df, user_input, top_n=5)

        st.session_state.history.append(("ユーザー", user_input))

        if result_df.empty:
            st.session_state.history.append(("AI", "まだ検索対象のQ&Aがありません。"))
        else:
            best = result_df.iloc[0]
            answer_text = (
                f"おすすめQ&A：\n\n"
                f"**校舎:** {best['campus']}\n\n"
                f"**質問:** {best['question']}\n\n"
                f"**回答:** {best['answer']}\n\n"
                f"**カテゴリ:** {best.get('category', '') or '-'}\n\n"
                f"**出典:** {best['source_type']} / {best.get('source_date', '-')}\n\n"
                f"**類似度:** {best['score']:.3f}"
            )
            st.session_state.history.append(("AI", answer_text))

        for role, msg in st.session_state.history:
            if role == "ユーザー":
                st.markdown(f"🧑‍💻 **あなた:** {msg}")
            else:
                st.markdown(f"🤖 **AI:** {msg}")

        if not result_df.empty:
            st.markdown("---")
            st.markdown("### 他にもこんなQ&Aがあります")
            for i in range(1, len(result_df)):
                row = result_df.iloc[i]
                st.markdown(
                    f"""
- **校舎:** {row['campus']}

  **Q:** {row['question']}

  **A:** {row['answer']}

  **カテゴリ:** {row.get('category', '') or '-'}  
  **出典:** {row['source_type']} / {row.get('source_date', '-')}  
  **類似度:** {row['score']:.3f}
                    """
                )

    except Exception as e:
        st.error(f"検索中にエラーが発生しました: {e}")

else:
    for role, msg in st.session_state.history:
        if role == "ユーザー":
            st.markdown(f"🧑‍💻 **あなた:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

st.markdown("---")
st.caption(
    "このアプリは Google Sheets と Google Docs からQ&Aを読み込み、"
    "最終更新から1時間経過したデータのみ検索対象にします。"
)
