# Heading-Based Privacy Policy Comparator (HTML-based, Side-by-Side)
import streamlit as st
import trafilatura
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fetch raw HTML from URL
def get_html_from_url(url):
    return trafilatura.fetch_url(url)

# Extract headings from HTML using tags like <h1> to <h6> and <strong>
def extract_headings_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong'])
    headings = [tag.get_text(strip=True) for tag in tags if 3 <= len(tag.get_text(strip=True)) <= 100]
    return list(set(headings))

# Match headings from policy A to policy B
# Return a side-by-side comparison with semantic similarity and color coding

def match_headings(headings_a, headings_b, threshold=0.75):
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

    # Add unmatched Policy B headings
    for idx, heading_b in enumerate(headings_b):
        if idx not in matched_indices:
            data.append({
                "Heading (Policy A)": "-",
                "Matched Heading (Policy B)": heading_b,
                "Similarity Score": "-",
                "Status": "❌ Missing in Policy A"
            })

    return data

# Streamlit UI
st.title("📋 Privacy Policy Heading Comparator")

st.markdown("Compare **headings** between two privacy policies and identify which sections are **missing or overlapping**.")

col1, col2 = st.columns(2)
with col1:
    url1 = st.text_input("Enter First Privacy Policy URL (e.g. NRL)")
with col2:
    url2 = st.text_input("Enter Second Privacy Policy URL (e.g. NFL)")

if st.button("Compare Policies") and url1 and url2:
    with st.spinner("Extracting and comparing headings..."):
        html_a = get_html_from_url(url1)
        html_b = get_html_from_url(url2)

        if html_a and html_b:
            headings_a = extract_headings_from_html(html_a)
            headings_b = extract_headings_from_html(html_b)

            comparison = match_headings(headings_a, headings_b)
            df = pd.DataFrame(comparison)
            st.success("✅ Comparison complete!")

            # Apply color styling
            def highlight_row(row):
                color = '#d1ffd1' if row['Status'] == '✔ Matched' else '#ffd6d6'
                return [f'background-color: {color}'] * len(row)

            styled_df = df.style.apply(highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Comparison CSV", csv, "comparison.csv", "text/csv")
        else:
            st.error("❌ Failed to extract HTML from one or both URLs.")