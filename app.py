import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import numpy as np
from urllib.parse import urlparse, unquote
from supabase import create_client, Client

# ========== Konfigurasi Supabase ==========
# === Konfigurasi Supabase ===
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== Judul Aplikasi ==========
st.set_page_config(page_title="Sistem Rekomendasi Buku", layout="wide")
st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Styling CSS ==========
st.markdown("""
<style>
    body {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', sans-serif;
    }
    .book-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .book-card:hover {
        transform: scale(1.02);
    }
    .book-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2b7aef;
        text-decoration: none;
    }
    .accuracy-tag {
        font-size: 0.85rem;
        color: #fff;
        background-color: #2b7aef;
        padding: 4px 10px;
        border-radius: 20px;
        float: right;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e90ff;
        margin-top: 20px;
        margin-bottom: 10px;
        border-left: 4px solid #1e90ff;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== Muat Data dari Supabase ==========
@st.cache_data(ttl=3600)  # Cache selama 1 jam
def load_data_from_supabase():
    try:
        response = client.table("katalog_buku").select("*").execute()
        df = pd.DataFrame(response.data)
        print(f"➡️ {len(df)} baris ditemukan di Supabase")
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

# ========== Fungsi Rekomendasi Berbasis Cosine Similarity ==========
def get_recommendations(query, top_n=5):
    query = query.strip().lower()
    if not query:
        return []

    matches = df[df["judul"].str.contains(query, case=False, na=False)]
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

# ========== Fungsi: Ekstrak ID dari URL ==========
def extract_id_from_url(url_path):
    path = urlparse(url_path).path
    parts = path.split("/")
    if "id" in parts:
        try:
            id_index = parts.index("id") + 1
            return int(parts[id_index])
        except (ValueError, IndexError):
            return None
    return None

# ========== Cari ID dari URL dan dapatkan judul ==========
query_params = st.query_params
url_path = query_params.get("url", "")

if url_path:
    book_id = extract_id_from_url(url_path)
    if book_id is not None and book_id in df['id'].values:
        selected_title = df[df['id'] == book_id]['judul'].iloc[0]
        st.success(f"🔍 Menemukan buku dengan ID `{book_id}` → _{selected_title}_")
    else:
        selected_title = ""
        st.warning("⚠️ ID tidak ditemukan dalam katalog.")
else:
    selected_title = ""

# ========== Today's Catalog Preview ==========
st.markdown('<div class="section-header">📖 Hari Ini di Katalog Buku</div>', unsafe_allow_html=True)
cols_today = st.columns(5)

random_indices = np.random.choice(len(df), size=min(5, len(df)), replace=False)

for col, i in zip(cols_today, random_indices):
    with col:
        row = df.iloc[i]
        img_url = get_book_image(row["url_katalog"])
        st.image(img_url, width=130)
        st.markdown(f"[{row['judul']}]({row['url_katalog']})", unsafe_allow_html=True)

st.markdown("---")

# ========== Input Pencarian Manual ==========
st.markdown('<div class="section-header">🔎 Cari Rekomendasi Buku</div>', unsafe_allow_html=True)

manual_query = st.text_input("Ketik sebagian judul buku...", placeholder="Contoh: Analisis")

if manual_query:
    filtered_titles = df[df["judul"].str.contains(manual_query, case=False, na=False)]["judul"].unique().tolist()
    if filtered_titles:
        selected_title = st.selectbox("Pilih judul:", options=filtered_titles[:50])

# ========== Tombol Cari Rekomendasi ==========
if st.button("🔎 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan pilih judul dari daftar.")
    else:
        with st.spinner("Memuat hasil..."):
            recommendations = get_recommendations(selected_title)

        if not recommendations:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            cols = st.columns(2)
            st.success(f"Rekomendasi untuk: _{selected_title}_")

            for i, book in enumerate(recommendations):
                with cols[i % 2]:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(book["gambar"], width=150)
                    st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    st.markdown(f"<span class='accuracy-tag'>{book['akurasi']}%</span>", unsafe_allow_html=True)
                    st.markdown('</div>')
                    st.markdown("---")

# ========== Footer ==========
st.markdown("<br><center>© Telkom University Library Recommendation System</center>", unsafe_allow_html=True)
