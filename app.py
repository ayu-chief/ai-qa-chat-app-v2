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
st.set_page_config(page_title="白井先生職員研修QA集", layout="centered")
st.title("白井先生職員研修QA集")
st.caption("Google Sheets と Google Docs の記録をまとめて検索できます")

SPREADSHEET_ID = "1hqwp1bvipMTPMiucddGz7_DtkE-dC9TAK4-BoD3yhrM"
DOC_ID = "1cDCIBNhPV37HeFT3Wtl6Dt18iz74mee5NsXbCJeuV5E"

DELAY_MINUTES = 0
CACHE_TTL_SECONDS = 300
JST = timezone(timedelta(hours=9))
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
    return bool(
        re.match(
            r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}(?:\s+\d{1,2}:\d{2}:\d{2})?$",
            text
        )
    )


def normalize_date_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("/", "-")
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return text


def clean_sheet_value(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for r in records:
        key = (
            r.get("source_type", ""),
            r.get("source_date", ""),
            r.get("sheet_name", ""),
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
def parse_sheet_rows(values: List[List[Any]], worksheet_title: str = "") -> List[Dict[str, Any]]:
    """
    シートの列位置や表記ゆれに強めに対応して Q/A を拾う。
    """
    records: List[Dict[str, Any]] = []

    current_date = ""
    pending_q = ""

    known_labels = {
        "京都", "名古屋", "木更津", "川崎", "尼崎", "盛岡", "湘南台", "町田", "大森", "その他"
    }

    def row_texts(row: List[Any]) -> List[str]:
        return [clean_sheet_value(str(x)) for x in row if clean_sheet_value(str(x))]

    for row in values:
        cells = row_texts(list(row))
        if not cells:
            continue

        found_date = None
        for cell in cells:
            if is_date_like(cell):
                found_date = normalize_date_text(cell)
                break
        if found_date:
            current_date = found_date
            pending_q = ""
            continue

        first = cells[0] if len(cells) > 0 else ""
        second = cells[1] if len(cells) > 1 else ""
        first_upper = first.upper()

        # 回答行
        if first_upper in {"A", "Ａ", "回答"}:
            answer = normalize_text(second)
            question = normalize_text(pending_q)
            if current_date and question and answer:
                records.append(
                    {
                        "source_type": "sheet",
                        "source_date": current_date,
                        "sheet_name": worksheet_title,
                        "category": "",
                        "question": question,
                        "answer": answer,
                    }
                )
            pending_q = ""
            continue

        a_match = re.match(r"^[AＡ][：:]\s*(.+)$", first)
        if a_match:
            answer = normalize_text(a_match.group(1))
            question = normalize_text(pending_q)
            if current_date and question and answer:
                records.append(
                    {
                        "source_type": "sheet",
                        "source_date": current_date,
                        "sheet_name": worksheet_title,
                        "category": "",
                        "question": question,
                        "answer": answer,
                    }
                )
            pending_q = ""
            continue

        a2_match = re.match(r"^回答[：:]\s*(.+)$", first)
        if a2_match:
            answer = normalize_text(a2_match.group(1))
            question = normalize_text(pending_q)
            if current_date and question and answer:
                records.append(
                    {
                        "source_type": "sheet",
                        "source_date": current_date,
                        "sheet_name": worksheet_title,
                        "category": "",
                        "question": question,
                        "answer": answer,
                    }
                )
            pending_q = ""
            continue

        # 質問行
        if first_upper in {"Q", "Ｑ", "質問"} and second:
            pending_q = normalize_text(second)
            continue

        q_match = re.match(r"^[QＱ][：:]\s*(.+)$", first)
        if q_match:
            pending_q = normalize_text(q_match.group(1))
            continue

        q2_match = re.match(r"^質問[：:]\s*(.+)$", first)
        if q2_match:
            pending_q = normalize_text(q2_match.group(1))
            continue

        # Q京都 / Q名古屋 / ...
        q_branch_match = re.match(r"^[QＱ]\s*.+$", first)
        if q_branch_match and second:
            pending_q = normalize_text(second)
            continue

        # 京都 / 名古屋 / ... + 2列目本文
        if first in known_labels and second:
            pending_q = normalize_text(second)
            continue

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
            "worksheet_counts": [],
            "raw_preview": [],
            "all_sheet_previews": {},
        }

    gc = get_gspread_client()
    sh = gc.open_by_key(SPREADSHEET_ID)

    all_records: List[Dict[str, Any]] = []
    worksheet_counts: List[Dict[str, Any]] = []
    raw_preview: List[List[Any]] = []
    all_sheet_previews: Dict[str, List[List[Any]]] = {}

    worksheets = sh.worksheets()
    batch_ranges = [f"'{ws.title}'!A:Z" for ws in worksheets]

    batch_response = sh.values_batch_get(batch_ranges)
    value_ranges = batch_response.get("valueRanges", [])

    for idx, ws in enumerate(worksheets):
        try:
            vr = value_ranges[idx] if idx < len(value_ranges) else {}
            values = vr.get("values", [])

            if idx == 0:
                raw_preview = values[:20]

            all_sheet_previews[ws.title] = values[:30]

            records = parse_sheet_rows(values, worksheet_title=ws.title)
            all_records.extend(records)

            worksheet_counts.append(
                {
                    "sheet_name": ws.title,
                    "row_count": len(values),
                    "qa_count": len(records),
                    "error": None,
                }
            )

        except Exception as e:
            worksheet_counts.append(
                {
                    "sheet_name": ws.title,
                    "row_count": 0,
                    "qa_count": 0,
                    "error": str(e),
                }
            )
            all_sheet_previews[ws.title] = []

    all_records = dedupe_records(all_records)

    return all_records, {
        "name": meta.get("name", "Sheets"),
        "modifiedTime": meta["modifiedTime"],
        "eligible": True,
        "reason": "",
        "worksheet_counts": worksheet_counts,
        "raw_preview": raw_preview,
        "all_sheet_previews": all_sheet_previews,
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
    current_question = ""
    current_category = ""
    answer_parts: List[str] = []

    def flush():
        nonlocal current_question, current_category, answer_parts
        q = normalize_text(current_question)
        a = normalize_text(" ".join(answer_parts))
        if q and a:
            records.append(
                {
                    "source_type": "doc",
                    "source_date": doc_date,
                    "sheet_name": "",
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

        if re.match(r"^■.+$", line):
            flush()
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
        df = pd.DataFrame(
            columns=[
                "source_type",
                "source_date",
                "sheet_name",
                "category",
                "question",
                "answer",
            ]
        )
    else:
        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)

    meta = {
        "sheet": sheet_meta,
        "doc": doc_meta,
        "count": len(df),
        "sheet_count": len(sheet_records),
        "doc_count": len(doc_records),
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
        df["question"].fillna("") + " "
        + df["answer"].fillna("") + " "
        + df["category"].fillna("")
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
        return df.iloc[0:0].copy()

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
    if not hits:
        return df.iloc[0:0].copy()

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

with st.spinner("検索モデルを準備中です。初回のみ少し時間がかかります..."):
    load_embedding_model()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("最新データに更新"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

with col2:
    show_admin = st.checkbox("詳細情報を表示", value=False)

if show_admin:
    with st.expander("データ反映状況", expanded=True):
        try:
            _, meta_preview = load_all_qa()
            st.write(f"合計件数: {meta_preview['count']} 件")
            st.write(f"Sheets抽出件数: {meta_preview['sheet_count']} 件")
            st.write(f"Docs抽出件数: {meta_preview['doc_count']} 件")

            st.markdown("**Sheets**")
            st.write(f"ファイル名: {meta_preview['sheet']['name']}")
            st.write(
                f"最終更新: {format_jst(parse_google_timestamp(meta_preview['sheet']['modifiedTime']))}"
            )
            st.write(f"取込対象: {'はい' if meta_preview['sheet']['eligible'] else 'まだ'}")
            if meta_preview["sheet"]["reason"]:
                st.caption(meta_preview["sheet"]["reason"])

            worksheet_counts = meta_preview["sheet"].get("worksheet_counts", [])
            if worksheet_counts:
                st.markdown("**Sheetsごとの抽出件数**")
                counts_df = pd.DataFrame(worksheet_counts)
                st.dataframe(counts_df, use_container_width=True)

                zero_df = counts_df[counts_df["qa_count"] == 0].copy()
                if not zero_df.empty:
                    st.markdown("**0件シートの中身確認**")
                    zero_sheet_names = zero_df["sheet_name"].tolist()
                    selected_zero_sheet = st.selectbox(
                        "0件だったシートを選んで先頭30行を表示",
                        zero_sheet_names,
                        key="zero_sheet_selector",
                    )

                    all_sheet_previews = meta_preview["sheet"].get("all_sheet_previews", {})
                    selected_preview = all_sheet_previews.get(selected_zero_sheet, [])
                    if selected_preview:
                        st.write(f"選択中シート: {selected_zero_sheet}")
                        st.dataframe(pd.DataFrame(selected_preview), use_container_width=True)
                    else:
                        st.write("このシートのプレビューは取得できませんでした。")

            st.markdown("**Docs**")
            st.write(f"ファイル名: {meta_preview['doc']['name']}")
            st.write(
                f"最終更新: {format_jst(parse_google_timestamp(meta_preview['doc']['modifiedTime']))}"
            )
            st.write(f"取込対象: {'はい' if meta_preview['doc']['eligible'] else 'まだ'}")
            if meta_preview["doc"]["reason"]:
                st.caption(meta_preview["doc"]["reason"])

            st.markdown("**Sheets raw preview（先頭シート20行）**")
            raw_preview = meta_preview["sheet"].get("raw_preview", [])
            if raw_preview:
                st.dataframe(pd.DataFrame(raw_preview), use_container_width=True)
            else:
                st.write("Sheetsのプレビューを取得できませんでした。")

        except Exception as e:
            st.error(f"データ状況の取得に失敗しました: {e}")

# ここだけ追加：注意書き
st.markdown(
    """
ここに書かれていることは教職員向けの言葉遣いになっています。  
実際に生徒・職員に対しての声がけの場面では直接的な表現にならないよう、ポジティブで柔らかい言葉遣いに変換してください。  
実際の行動に移るとき（声がけ・保護者連絡等のアクション）は回答内容を参考に、校舎内でコミュニケーションをとったうえで活用するようにしてください。
    """
)

with st.form(key="chat_form", clear_on_submit=False):
    user_input = st.text_input("知りたいこと・悩みを入力してください", key="user_input")
    st.markdown(
        """
##### 入力例
- 例1：「不登校」
- 例2：「友人とのトラブルがあったときの対応は？」
- 例3：「保護者対応で境界線をどう引く？」
        """
    )
    search_btn = st.form_submit_button("検索")

if search_btn and user_input:
    try:
        with st.spinner("検索中..."):
            time.sleep(0.2)
            df, _meta = load_all_qa()
            result_df = search_qa(df, user_input, top_n=5)

        st.session_state.history.append(("ユーザー", user_input))

        if result_df.empty:
            st.session_state.history.append(
                (
                    "AI",
                    "まだ検索対象のQ&Aがありません。少し待ってから『最新データに更新』を押すか、入力語句を変えて試してください。",
                )
            )
        else:
            best = result_df.iloc[0]
            answer_text = (
                f"おすすめQ&A：\n\n"
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
- **Q:** {row['question']}

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
st.caption("このアプリは Google Sheets と Google Docs からQ&Aを読み込み、意味検索で近い内容を表示します。")
