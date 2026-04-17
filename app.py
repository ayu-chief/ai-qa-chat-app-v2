import re
import time
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
    layout="centered",
)


# =========================
# CSS
# =========================
st.markdown("""
<style>
.main {background-color:#f7f9fc;}

.hero {
    background:#eef4ff;
    padding:20px;
    border-radius:14px;
    margin-bottom:15px;
}

.card {
    background:#ffffff;
    padding:16px;
    border-radius:12px;
    margin-bottom:12px;
    border:1px solid #eee;
}

.best {
    border-left:6px solid #4A90E2;
}

.notice-text {
    font-size:0.9rem;
    color:#555;
    line-height:1.8;
    margin-bottom:18px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# タイトル（囲いあり）
# =========================
st.markdown("""
<div class="hero">
<h2>白井先生職員研修QA集</h2>
<p>相談内容を入力すると、類似する対応事例を表示します</p>
</div>
""", unsafe_allow_html=True)


# =========================
# 注意書き（囲いなし）
# =========================
st.markdown("""
<div class="notice-text">
ここに書かれていることは教職員向けの言葉遣いになっています。<br>
実際に生徒・職員に対しての声がけの場面では直接的な表現にならないよう、ポジティブで柔らかい言葉遣いに変換してください。<br>
実際の行動に移るとき（声がけ・保護者連絡等のアクション）は回答内容を参考に、校舎内でコミュニケーションをとったうえで活用するようにしてください。
</div>
""", unsafe_allow_html=True)


# =========================
# 認証
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
# Sheets
# =========================
def load_sheets():
    gc = gspread.authorize(get_creds())
    sh = gc.open_by_key("1hqwp1bvipMTPMiucddGz7_DtkE-dC9TAK4-BoD3yhrM")

    data = []

    for ws in sh.worksheets():
        rows = ws.get_all_values()
        q = ""
        date = ""

        for r in rows:
            r = [x for x in r if x]

            if not r:
                continue

            if re.match(r"\d{4}", r[0]):
                date = r[0]
                continue

            if r[0].startswith("Q") and len(r) > 1:
                q = r[1]
                continue

            if r[0] in ["A", "回答"] and len(r) > 1:
                data.append({
                    "question": q,
                    "answer": r[1],
                    "date": date,
                    "source": "sheet"
                })

    return data


# =========================
# Docs
# =========================
def load_docs():
    service = build("docs", "v1", credentials=get_creds())
    doc = service.documents().get(documentId="1cDCIBNhPV37HeFT3Wtl6Dt18iz74mee5NsXbCJeuV5E").execute()

    text = ""
    for c in doc["body"]["content"]:
        if "paragraph" in c:
            for e in c["paragraph"]["elements"]:
                if "textRun" in e:
                    text += e["textRun"]["content"]

    lines = text.split("\n")

    data = []
    q = ""

    for line in lines:
        line = line.strip()

        if line.startswith("質問"):
            q = line.replace("質問:", "")
        elif line.startswith("回答"):
            data.append({
                "question": q,
                "answer": line.replace("回答:", ""),
                "date": "",
                "source": "doc"
            })

    return data


# =========================
# 検索
# =========================
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def search(df, query):
    model = get_model()

    corpus = (df["question"] + " " + df["answer"]).tolist()
    emb = model.encode(corpus, convert_to_tensor=True)
    q_emb = model.encode([query], convert_to_tensor=True)

    hits = util.semantic_search(q_emb, emb)[0][:5]
    return df.iloc[[h["corpus_id"] for h in hits]]


# =========================
# データ
# =========================
@st.cache_data
def load_all():
    return pd.DataFrame(load_sheets() + load_docs())


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

    if not result.empty:

        # ★おすすめ1件
        best = result.iloc[0]
        st.markdown("### ⭐ おすすめのQ&A")

        st.markdown(f"""
        <div class="card best">
        <b>Q:</b> {best['question']}<br><br>
        <b>A:</b> {best['answer']}
        </div>
        """, unsafe_allow_html=True)

        # ★他候補
        st.markdown("### 他の候補")

        for i in range(1, len(result)):
            row = result.iloc[i]

            st.markdown(f"""
            <div class="card">
            <b>Q:</b> {row['question']}<br><br>
            <b>A:</b> {row['answer']}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.write("該当するQ&Aが見つかりませんでした")
