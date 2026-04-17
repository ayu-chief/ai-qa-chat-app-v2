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
# 基本設定（★タイトル変更）
# =========================
st.set_page_config(
    page_title="白井先生職員研修QA集",
    page_icon="📘",
    layout="centered",
)

SPREADSHEET_ID = "1hqwp1bvipMTPMiucddGz7_DtkE-dC9TAK4-BoD3yhrM"
DOC_ID = "1cDCIBNhPV37HeFT3Wtl6Dt18iz74mee5NsXbCJeuV5E"

JST = timezone(timedelta(hours=9))
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# UIデザイン
# =========================
st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.hero-box {background:#eef4ff;border-radius:16px;padding:20px;margin-bottom:20px;}
.hero-title {font-size:1.8rem;font-weight:800;}
.hero-sub {color:#555;}
.card {background:#fff;padding:15px;border-radius:12px;margin-bottom:12px;border:1px solid #eee;}
</style>
""", unsafe_allow_html=True)


# =========================
# ヘッダー（★タイトル変更）
# =========================
st.markdown("""
<div class="hero-box">
<div class="hero-title">📘 白井先生職員研修QA集</div>
<div class="hero-sub">相談内容を入力すると、類似する対応事例を表示します</div>
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
# Sheets読み込み
# =========================
def load_sheets():
    gc = gspread.authorize(get_creds())
    sh = gc.open_by_key(SPREADSHEET_ID)

    all_records = []

    for ws in sh.worksheets():
        values = ws.get_all_values()

        date = ""
        q = ""

        for row in values:
            row = [str(x).strip() for x in row if str(x).strip()]

            if not row:
                continue

            if re.match(r"\d{4}/\d{1,2}/\d{1,2}", row[0]):
                date = row[0]
                continue

            if row[0].startswith("Q"):
                if len(row) > 1:
                    q = row[1]
                continue

            if row[0] == "質問" and len(row) > 1:
                q = row[1]
                continue

            if row[0] in ["A", "回答"] and len(row) > 1:
                all_records.append({
                    "question": q,
                    "answer": row[1],
                    "source": "sheet",
                    "date": date
                })

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

        if line.startswith("質問"):
            q = line.replace("質問:", "")
        elif line.startswith("回答"):
            records.append({
                "question": q,
                "answer": line.replace("回答:", ""),
                "source": "doc",
                "date": ""
            })

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
    return df.iloc[[h["corpus_id"] for h in hits]]


# =========================
# データロード
# =========================
@st.cache_data
def load_all():
    data = load_sheets() + load_docs()
    return pd.DataFrame(data)


# =========================
# UI
# =========================
col1, col2 = st.columns([4,1])

with col1:
    user_input = st.text_input("", placeholder="例：不登校、保護者対応、友人トラブル")

with col2:
    search_btn = st.button("検索")


# =========================
# 検索実行
# =========================
if search_btn and user_input:
    df = load_all()

    with st.spinner("検索中..."):
        result = search(df, user_input)

    st.markdown("### 🔍 検索結果")

    for _, row in result.iterrows():
        st.markdown(f"""
        <div class="card">
        <b>Q:</b> {row['question']}<br><br>
        <b>A:</b> {row['answer']}<br><br>
        <small>{row['source']} / {row['date']}</small>
        </div>
        """, unsafe_allow_html=True)


# =========================
# フッター（★注意書き追加）
# =========================
st.markdown("""
---
<div style="font-size:0.85rem; color:#666; line-height:1.8">

このアプリは Google Sheets と Google Docs の記録をもとに検索を行います。<br><br>

ここに書かれていることは教職員向けの言葉遣いになっています。<br>
実際に生徒・職員に対しての声がけの場面では直接的な表現にならないよう、ポジティブで柔らかい言葉遣いに変換してください。<br>
実際の行動に移るとき（声がけ・保護者連絡等のアクション）は回答内容を参考に、校舎内でコミュニケーションをとったうえで活用するようにしてください。

</div>
""", unsafe_allow_html=True)
