from pathlib import Path
import os, json, time

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

DATA = Path("data")
st.set_page_config(page_title="H&M Recommendations", layout="wide")


# ---------- Load data once ----------
@st.cache_resource
def load_data():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    arts = con.execute(f"""
        SELECT article_id, prod_name, product_type_name, colour_group_name,
               index_group_name, COALESCE(detail_desc, '') AS detail_desc
        FROM '{(DATA / 'articles.parquet').as_posix()}' ORDER BY article_id
    """).df()
    recs = pd.read_parquet(DATA / "app_recs.parquet")
    labels = pd.read_parquet(DATA / "labels_test.parquet")
    hist = con.execute(f"""
        SELECT customer_id, article_id, MAX(t_dat) AS last_buy
        FROM '{(DATA / 'sample_transactions.parquet').as_posix()}'
        WHERE w BETWEEN 1 AND 8
        GROUP BY ALL
    """).df()
    available = set(con.execute(f"""
        SELECT DISTINCT article_id FROM '{(DATA / 'transactions_train.parquet').as_posix()}'
        WHERE t_dat > DATE '2020-09-22' - INTERVAL 56 DAY
    """).df()["article_id"])
    return arts, recs, labels, hist, available


@st.cache_resource
def load_search(_arts, available):
    from sentence_transformers import SentenceTransformer
    emb = np.load(DATA / "article_emb.npy")
    ids = np.load(DATA / "article_emb_ids.npy")
    assert (_arts["article_id"].to_numpy() == ids).all(), "Row order mismatch"
    mask = _arts["article_id"].isin(available).to_numpy()
    cat = _arts[mask].reset_index(drop=True)
    cat["product_code"] = cat["article_id"].str[:7]
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return cat, emb[mask], model


arts, recs, labels, hist, available = load_data()
info = arts.set_index("article_id")
bought_test = {c: set(b) for c, b in zip(labels["customer_id"], labels["bought"])}


def describe(ids):
    df = info.reindex(ids)[["prod_name", "product_type_name", "colour_group_name", "index_group_name"]]
    df.columns = ["Product", "Type", "Colour", "Section"]
    return df.reset_index(names="article_id")


# ---------- Page ----------
st.title("H&M personalised recommendations")
st.caption("Portfolio project. 20% customer sample, last 26 weeks of 2020 data. Recommendations are for the test week, 16 to 22 September 2020.")

tab1, tab2 = st.tabs(["Recommendations", "AI product search"])

# ---------- Tab 1: recommendations ----------
with tab1:
    with_hist = sorted(set(recs["customer_id"]) & set(hist["customer_id"]))[:300]
    cid = st.selectbox(
        "Pick a customer (300 test-week buyers who also shopped in the 8 weeks before)",
        with_hist, format_func=lambda c: c[:12] + "…",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Bought in the 8 weeks before")
        past = hist[hist["customer_id"] == cid].sort_values("last_buy", ascending=False)
        p = describe(past["article_id"].tolist())
        p.insert(0, "Last bought", past["last_buy"].dt.date.to_numpy())
        st.dataframe(p.drop(columns="article_id"), hide_index=True, use_container_width=True)

    with right:
        st.subheader("12 recommendations (LightGBM ranker)")
        rec_ids = list(recs.loc[recs["customer_id"] == cid, "recs"].iloc[0])
        r = describe(rec_ids)
        r.insert(0, "Rank", range(1, len(r) + 1))
        r["Bought in test week"] = ["✅" if a in bought_test.get(cid, set()) else "" for a in rec_ids]
        st.dataframe(r.drop(columns="article_id"), hide_index=True, use_container_width=True)
        hits = sum(a in bought_test.get(cid, set()) for a in rec_ids)
        st.metric("Recommendations this customer actually bought", f"{hits} of 12")

    st.info("Most customers buy none of their 12 picks. Across all 13,637 test customers the ranker's MAP@12 was 0.02818, "
            "3.5 times the popularity baseline. Low numbers are normal for this task.")

# ---------- Tab 2: AI search ----------
SYSTEM = """You are a shopping assistant for H&M.
You may ONLY recommend products from the PRODUCTS list you are given.
Never invent products, names, colours, prices or details that are not in the list.
Pick up to 5 products that best match the request.
Leave out products that only partly match (wrong colour, wrong type, wrong section such as kids vs men).
If nothing in the list matches, say so honestly and recommend nothing.
Reply as JSON with two keys:
"answer": 2 to 4 short, friendly sentences for the customer, naming the products you picked.
"article_ids": the article_id values you recommend, copied exactly from the list."""


def retrieve(query, cat, cat_emb, model, k=20):
    q = model.encode([query], normalize_embeddings=True)[0]
    out, seen = [], set()
    for i in np.argsort(-(cat_emb @ q)):
        pc = cat.at[i, "product_code"]
        if pc in seen:
            continue
        seen.add(pc)
        row = cat.iloc[i]
        out.append({"article_id": row["article_id"], "name": row["prod_name"], "type": row["product_type_name"],
                    "colour": row["colour_group_name"], "section": row["index_group_name"],
                    "desc": row["detail_desc"][:200]})
        if len(out) == k:
            break
    return out


def ask_gemini(query, items):
    from google import genai
    from google.genai import types
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                          http_options=types.HttpOptions(timeout=60_000))
    prompt = f"REQUEST: {query}\n\nPRODUCTS:\n{json.dumps(items, ensure_ascii=False)}"
    models = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    wait = 3
    for attempt in range(4):
        name = models[attempt % 2]
        try:
            resp = client.models.generate_content(
                model=name, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM,
                                                   response_mime_type="application/json", temperature=0))
            return json.loads(resp.text), name
        except Exception:
            time.sleep(wait)
            wait *= 2
    return None, None


with tab2:
    st.write("Describe what you want. The app finds the 20 closest products by meaning, "
             "then Gemini picks the best ones from only those 20.")
    query = st.text_input("What are you looking for?", placeholder="e.g. warm winter jacket for men")

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Loading the search model (first time only) and finding products..."):
            cat, cat_emb, model = load_search(arts, available)
            items = retrieve(query, cat, cat_emb, model)

        with st.spinner("Asking Gemini..."):
            out, used = ask_gemini(query, items)

        by_id = {it["article_id"]: it for it in items}
        if out is None:
            st.warning("Gemini is busy right now, so here are the closest products from retrieval only.")
        else:
            ids = out.get("article_ids", [])
            invented = [a for a in ids if a not in by_id]
            st.success(out.get("answer", ""))
            st.caption(f"Answered by {used}. Grounding check: "
                       + ("all picks came from the retrieved list." if not invented
                          else f"{len(invented)} pick(s) were NOT in the list and were removed."))
            picks = [a for a in ids if a in by_id]
            if picks:
                st.dataframe(describe(picks).drop(columns="article_id"), hide_index=True, use_container_width=True)

        with st.expander("See all 20 retrieved products"):
            st.dataframe(describe([it["article_id"] for it in items]).drop(columns="article_id"),
                         hide_index=True, use_container_width=True)