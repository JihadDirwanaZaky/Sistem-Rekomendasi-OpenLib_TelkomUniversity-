import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from supabase import create_client, Client

# ========== Konfigurasi Supabase ==========
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"


client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== Judul Aplikasi ==========
st.set_page_config(page_title="Sistem Rekomendasi Buku", layout="wide")
st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Muat Data dari Supabase ==========
@st.cache_data(ttl=3600)
def load_data_from_supabase():
    try:
        response = client.table("katalog_buku").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"❌ Error mengambil data dari Supabase: {e}")
        st.stop()

df = load_data_from_supabase()

# Pastikan kolom penting ada
required_columns = ["judul", "url_katalog", "combined_text"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan di database Supabase")
        st.stop()

# ========== Setup TF-IDF dan Cosine Similarity ==========
@st.cache_resource
def setup_tfidf():
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, cosine_sim

vectorizer, tfidf_matrix, cosine_sim = setup_tfidf()

# ========== Fungsi untuk Ambil Gambar ==========
def get_book_image(url_katalog):
    try:
        response = requests.get(url_katalog, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', class_='thumbnail') or soup.find('img', class_='cover')
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
        else:
            return "https://via.placeholder.com/150x220?text=No+Image "
    except Exception as e:
        return "https://via.placeholder.com/150x220?text=No+Image "

# ========== Fungsi Rekomendasi Berbasis ID ==========
def get_recommendations_by_id(book_id, top_n=5):
    matches = df[df["id"] == book_id]
    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    filtered_scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for index, score in filtered_scores:
        results.append({
            "judul": df.iloc[index]["judul"],
            "url_katalog": df.iloc[index]["url_katalog"],
            "gambar": get_book_image(df.iloc[index]["url_katalog"]),
            "akurasi": round(score * 100, 2)
        })

    return results

# ========== Parsing ID dari URL ==========
def extract_id_from_url(url_path):
    parsed = urlparse(url_path)
    path_parts = parsed.path.split("/")
    if "id" in path_parts:
        try:
            id_index = path_parts.index("id") + 1
            return int(path_parts[id_index])
        except (ValueError, IndexError):
            return None
    return None

# ========== Cari ID dari Parameter URL ==========
query_params = st.query_params
url_path = query_params.get("url", "")

book_id = None
if url_path:
    book_id = extract_id_from_url(url_path)
else:
    st.info("🔍 Silakan gunakan format berikut untuk quick search:")
    st.code("https://openlibraryrecommend.streamlit.app/?url=https ://openlibrary.telkomuniversity.ac.id/home/catalog/id/232959/slug/...", language="text")

# ========== Tampilkan Hasil Rekomendasi Otomatis ==========
if book_id is not None and book_id in df["id"].values:
    selected_title = df[df["id"] == book_id]["judul"].iloc[0]
    with st.spinner("Memuat rekomendasi..."):
        recommendations = get_recommendations_by_id(book_id)
elif url_path and (book_id is None or book_id not in df["id"].values):
    st.warning("⚠️ ID buku tidak ditemukan dalam katalog.")
    recommendations = []
else:
    selected_title = ""
    recommendations = []

# ========== UI Rekomendasi ==========
if url_path and book_id in df["id"].values:
    st.success(f"Rekomendasi untuk: _{selected_title}_")

    cols = st.columns(2)
    for i, book in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown('<div class="book-card">', unsafe_allow_html=True)
            st.image(book["gambar"], width=150)
            st.markdown(f"[{book['judul']}]({book['url_katalog']})", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.85rem; color:#777;'>Akurasi: {book['akurasi']}%</span>", unsafe_allow_html=True)
            st.markdown("</div>")
            st.markdown("---")

# ========== Manual Search ==========
else:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("🔍 Atau cari manual menggunakan judul buku:")

    query = st.text_input("Ketik sebagian judul buku...")
    if query:
        filtered_titles = df[df["judul"].str.contains(query, case=False, na=False)]["judul"].unique().tolist()
        if filtered_titles:
            selected_title = st.selectbox("Pilih judul:", options=filtered_titles[:50])
        else:
            st.info("Tidak ada judul yang cocok.")

        if st.button("🔎 Cari Rekomendasi"):
            if selected_title:
                idx = df[df["judul"] == selected_title].index[0]
                recommendations = get_recommendations_by_id(df.loc[idx, "id"])
                cols = st.columns(2)
                for i, book in enumerate(recommendations):
                    with cols[i % 2]:
                        st.markdown('<div class="book-card">', unsafe_allow_html=True)
                        st.image(book["gambar"], width=150)
                        st.markdown(f"[{book['judul']}]({book['url_katalog']})", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-size:0.85rem; color:#777;'>Akurasi: {book['akurasi']}%</span>", unsafe_allow_html=True)
                        st.markdown("</div>")
                        st.markdown("---")
