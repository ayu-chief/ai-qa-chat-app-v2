import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer, util


# =========================
# 基本設定
# =========================
st.set_page_config(
    page_title="白井先生職員研修QA集",
    page_icon="📘",
    layout="centered",
)

SPREADSHEET_ID = "1hqwp1bvipMTPMiucddGz7_DtkE-dC9TAK4-BoD3yhrM"
DOC_ID = "1cDCIBNhPV37HeFT3Wtl6Dt18iz74mee5NsXbCJeuV5E"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# UIデザイン
# =========================
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.hero-box {
    background: #eef4ff;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 14px;
    border: 1px solid #d9e6fb;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #1f2a44;
    margin-bottom: 6px;
}
.hero-sub {
    color: #5a6477;
    line-height: 1.7;
}
.notice-box {
    background: #fff8e8;
    border: 1px solid #f0dba8;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 18px;
    color: #5f512d;
    line-height: 1.8;
    font-size: 0.93rem;
}
.card {
    background: #ffffff;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 14px;
    border: 1px solid #e8edf5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.result-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 18px 0 10px 0;
    color: #24324a;
}
div[data-testid="stTextInput"] input {
    border-radius: 12px;
}
div[data-testid="stButton"] > button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# =========================
# ヘッダー
# =========================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">白井先生職員研修QA集</div>
    <div class="hero-sub">相談内容を入力すると、類似する対応事例を表示します</div>
</div>
""", unsafe_allow_html=True)

# =========================
# 注意書き（検索欄の上へ移動）
# =========================
st.markdown("""
<div class="notice-box">
ここに書かれていることは教職員向けの言葉遣いになっています。<br>
実際に生徒・職員に対しての声がけの場面では直接的な表現にならないよう、ポジティブで柔らかい言葉遣いに変換してください。<br>
実際の行動に移るとき（声がけ・保護者連絡等のアクション）は回答内容を参考に、校舎内でコミュニケーションをとったうえで活用するようにしてください。
</div>
""", unsafe_allow_html=True)


# =========================
# Google認証
# =========================
def get_creds():
    return Credentials.from_service_account_info(
        st.secrets["google_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/documents.readonly",
        ],
    )


# =========================
# 共通
# =========================
def normalize_text(v: Any) -> str:
    return str(v).replace("\u3000", " ").strip() if v is not None else ""


def esc_html(text: Any) -> str:
    text = normalize_text(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# =========================
# Sheets読み込み
# =========================
def load_sheets():
    gc = gspread.authorize(get_creds())
    sh = gc.open_by_key(SPREADSHEET_ID)

    all_records = []

    for ws in sh.worksheets():
        values = ws.get_all_values()

        current_date = ""
        q = ""

        for row in values:
            row = [normalize_text(x) for x in row if normalize_text(x)]
            if not row:
                continue

            # 日付行
            if re.match(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", row[0]):
                current_date = row[0].replace("/", "-")
                continue

            first = row[0]
            second = row[1] if len(row) > 1 else ""

            # 質問行
            if first.startswith("Q") and second:
                q = second
                continue

            if first == "質問" and second:
                q = second
                continue

            q_match = re.match(r"^質問[:：]\s*(.+)$", first)
            if q_match:
                q = q_match.group(1)
                continue

            # 回答行
            if first in ["A", "回答"] and second:
                if q:
                    all_records.append({
                        "question": q,
                        "answer": second,
                        "source": "sheet",
                        "date": current_date,
                    })
                q = ""
                continue

            a_match = re.match(r"^回答[:：]\s*(.+)$", first)
            if a_match:
                if q:
                    all_records.append({
                        "question": q,
                        "answer": a_match.group(1),
                        "source": "sheet",
                        "date": current_date,
                    })
                q = ""

    return all_records


# =========================
# Docs読み込み
# =========================
def load_docs():
    service = build("docs", "v1", credentials=get_creds())
    doc = service.documents().get(documentId=DOC_ID).execute()

    text = ""
    for c in doc["body"]["content"]:
        if "paragraph" in c:
            for e in c["paragraph"]["elements"]:
                if "textRun" in e:
                    text += e["textRun"]["content"]

    lines = text.split("\n")

    records = []
    q = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("質問"):
            q = line.replace("質問:", "").replace("質問：", "").strip()
            continue

        if line.startswith("回答"):
            a = line.replace("回答:", "").replace("回答：", "").strip()
            if q and a:
                records.append({
                    "question": q,
                    "answer": a,
                    "source": "doc",
                    "date": "",
                })
                q = ""

    return records


# =========================
# 検索
# =========================
@st.cache_resource
def get_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def search(df, query):
    model = get_model()
    corpus = (df["question"] + " " + df["answer"]).tolist()
    emb = model.encode(corpus, convert_to_tensor=True)
    q_emb = model.encode([query], convert_to_tensor=True)
    hits = util.semantic_search(q_emb, emb)[0][:5]
    return df.iloc[[h["corpus_id"] for h in hits]].copy()


# =========================
# データロード
# =========================
@st.cache_data
def load_all():
    data = load_sheets() + load_docs()
    return pd.DataFrame(data)


# =========================
# 検索入力UI
# =========================
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "",
        placeholder="例：不登校、保護者対応、友人トラブル",
    )

with col2:
    search_btn = st.button("検索", use_container_width=True)


# =========================
# 検索実行
# =========================
if search_btn and user_input:
    df = load_all()

    with st.spinner("検索中..."):
        result = search(df, user_input)

    st.markdown('<div class="result-title">検索結果</div>', unsafe_allow_html=True)

    for _, row in result.iterrows():
        q = esc_html(row["question"])
        a = esc_html(row["answer"])
        src = esc_html(row["source"])
        dt = esc_html(row["date"])

        st.markdown(f"""
        <div class="card">
            <b>Q:</b> {q}<br><br>
            <b>A:</b> {a}<br><br>
            <small>{src} / {dt}</small>
        </div>
        """, unsafe_allow_html=True)
