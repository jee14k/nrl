import streamlit as st
import re
import requests
import trafilatura
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Load SerpAPI key
SERPAPI_API_KEY = "8ce91dfbc822680fc4fbcca6ac148b05fd51d8a47225fd75d94102510924f721"

# Load embedding model
model = SentenceTransformer("paraphrase-mpnet-base-v2", device="cpu")

# -------- Helper Functions --------

def normalize_heading(text):
    text = re.sub(r'^[0-9.\s\-:()]+', '', text)  # remove numbering
    text = text.lower()
    text = re.sub(r'\b(nrl|nbl)\b', '', text)  # remove league name
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    return text.strip()

def serpapi_search_privacy_policy(league_name):
    if not SERPAPI_API_KEY:
        return None
    query = f"{league_name} privacy policy"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 5
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        for result in data.get("organic_results", []):
            link = result.get("link", "")
            if "privacy" in link.lower() or "policy" in link.lower():
                return link
        return None
    except Exception as e:
        st.error(f"SerpAPI error: {e}")
        return None

def get_html_from_url(url):
    return trafilatura.fetch_url(url)

def extract_headings_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong"])
    headings = [tag.get_text(strip=True) for tag in tags if 3 <= len(tag.get_text(strip=True)) <= 100]
    return list(set(headings))

def match_headings(headings_a, headings_b, threshold=0.65):
    data = []
    embeddings_a = model.encode(headings_a, convert_to_tensor=True)
    embeddings_b = model.encode(headings_b, convert_to_tensor=True)

    matched_indices = set()

    for i, heading_a in enumerate(headings_a):
        similarities = util.cos_sim(embeddings_a[i], embeddings_b)[0]
        best_score = float(similarities.max())
        best_idx = int(similarities.argmax())
        match_b = headings_b[best_idx] if best_score >= threshold else "Not Found"

        if best_score >= threshold:
            matched_indices.add(best_idx)

        data.append({
            "Heading (Policy A)": heading_a,
            "Matched Heading (Policy B)": match_b,
            "Similarity Score": f"{best_score:.2f}" if best_score >= threshold else "-",
            "Status": "✔ Matched" if best_score >= threshold else "❌ Missing in Policy B"
        })

    for idx, heading_b in enumerate(headings_b):
        if idx not in matched_indices:
            data.append({
                "Heading (Policy A)": "-",
                "Matched Heading (Policy B)": heading_b,
                "Similarity Score": "-",
                "Status": "❌ Missing in Policy A"
            })

    return data

# -------- Streamlit UI --------

st.title("📋 Privacy Policy Comparator")

st.markdown(
    "Compare **headings** between two privacy policies using **SerpAPI** to automatically find official URLs."
)

st.subheader("🔎 AI-Assisted URL Finder")

col1, col2 = st.columns(2)
with col1:
    league1 = st.text_input("Enter Sports League 1 (e.g. NRL)", key="league1")
with col2:
    league2 = st.text_input("Enter Sports League 2 (e.g. NFL)", key="league2")

if st.button("Find URLs"):
    if not league1 or not league2:
        st.warning("Please enter both league names.")
    else:
        with st.spinner("Searching SerpAPI for policy URLs..."):
            url1 = serpapi_search_privacy_policy(league1)
            url2 = serpapi_search_privacy_policy(league2)

            st.session_state["autofill_url1"] = url1 if url1 else ""
            st.session_state["autofill_url2"] = url2 if url2 else ""

            st.write(f"🔗 {league1} Privacy Policy: {url1 or 'Not found'}")
            st.write(f"🔗 {league2} Privacy Policy: {url2 or 'Not found'}")

st.subheader("🔗 Policy URLs")

col3, col4 = st.columns(2)
with col3:
    url1 = st.text_input("Policy URL 1", value=st.session_state.get("autofill_url1", ""))
with col4:
    url2 = st.text_input("Policy URL 2", value=st.session_state.get("autofill_url2", ""))

if st.button("Compare Policies") and url1 and url2:
    with st.spinner("Comparing privacy policy headings..."):
        html_a = get_html_from_url(url1)
        html_b = get_html_from_url(url2)

        if html_a and html_b:
            headings_a_raw = extract_headings_from_html(html_a)
            headings_b_raw = extract_headings_from_html(html_b)

            # Normalize headings
            headings_a = [normalize_heading(h) for h in headings_a_raw]
            headings_b = [normalize_heading(h) for h in headings_b_raw]

            comparison = match_headings(headings_a, headings_b)
            df = pd.DataFrame(comparison)
            st.success("✅ Comparison complete!")

            def highlight_row(row):
                color = "#d1ffd1" if row["Status"] == "✔ Matched" else "#ffd6d6"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "comparison.csv", "text/csv")
        else:
            st.error("❌ Could not fetch or parse one or both URLs.")
