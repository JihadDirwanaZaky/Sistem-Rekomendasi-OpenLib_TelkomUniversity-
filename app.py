import streamlit as st
import pandas as pd
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh

# ========== Judul Aplikasi ==========
st.set_page_config(page_title="Sistem Rekomendasi Buku", layout="wide")
st.markdown("""
<style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .book-card {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 15px;
    }
    .book-card:hover {
        transform: scale(1.03);
    }
    .book-title {
        font-size: 1rem;
        font-weight: bold;
        color: #2c3e50;
        text-decoration: none;
    }
    .accuracy {
        color: gray;
        font-size: 0.9rem;
    }
    .rotating-container {
        display: flex;
        gap: 10px;
        overflow-x: auto;
        padding-bottom: 10px;
    }
    .rotating-card {
        flex: 0 0 auto;
        width: 150px;
        text-align: center;
    }
    .rotating-card img {
        border-radius: 4px;
        width: 100%;
        height: auto;
    }
    .section-header {
        background-color: #2c3e50;
        color: white;
        padding: 8px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Muat Data ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("katalog_bersih_pre-processed_ulang.csv", nrows=50000)
    except Exception as e:
        st.error(f"❌ Error membaca file CSV: {e}")
        st.stop()

    required_columns = ["judul", "judul_clean", "combined_text", "url_katalog", "klasifikasi", "jenis", "subjek"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Kolom '{col}' tidak ditemukan dalam file CSV")
            st.stop()

    df.dropna(subset=["judul_clean", "combined_text", "url_katalog"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

# ========== TF-IDF Setup ==========
@st.cache_resource
def setup_tfidf():
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, cosine_sim

vectorizer, tfidf_matrix, cosine_sim = setup_tfidf()

# ========== Fungsi untuk Mengambil Gambar Buku ==========
def get_book_image(url_katalog):
    try:
        response = requests.get(url_katalog, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', class_='thumbnail') or soup.find('img', class_='cover')
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
        else:
            return "https://via.placeholder.com/150x220?text=No+Image"
    except:
        return "https://via.placeholder.com/150x220?text=No+Image"

# ========== Cari Rekomendasi ==========
def get_recommendations(idx, top_n=5, include_self=False):
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if not include_self:
        scores = [s for s in scores if s[0] != idx]
    top_scores = scores[:top_n]
    results = []
    for i, score in top_scores:
        results.append({
            "judul": df.iloc[i]["judul"],
            "url_katalog": df.iloc[i]["url_katalog"],
            "gambar": get_book_image(df.iloc[i]["url_katalog"]),
            "akurasi": round(score * 100, 2)
        })
    return results

# ========== Rotating Container (Try These Catalogs) ==========
# Autorefresh setiap 15 detik (15000 ms)
st_autorefresh(interval=15000, limit=None, key="rotating_refresh")

st.header("📖 Daftar Acak di Katalog (Berubah tiap 15 detik)")
random_indices = random.sample(range(len(df)), 5)
cols_today = st.container()
with cols_today:
    st.markdown('<div class="rotating-container">', unsafe_allow_html=True)
    for i in random_indices:
        row = df.iloc[i]
        img_url = get_book_image(row["url_katalog"])
        st.markdown(f"""
            <div class="rotating-card">
                <a href="{row["url_katalog"]}" target="_blank">
                    <img src="{img_url}" alt="Cover">
                    <div style="margin-top:5px; font-size:0.9rem; color:#2c3e50;">{row["judul"][:25]}{"..." if len(row["judul"])>25 else ""}</div>
                </a>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ========== UI Streamlit – Pencarian ==========
search_by = st.selectbox("🔍 Search by:", ["Title", "URL"])

if search_by == "Title":
    query_input = st.text_input("Ketik sebagian judul buku...", placeholder="Contoh: Analisis")
    filtered_titles = []
    if query_input:
        filtered_titles = df[df["judul_clean"].str.contains(query_input.strip(), case=False, na=False)]["judul"].unique().tolist()
    selected_title = st.selectbox("Pilih judul buku:", filtered_titles) if filtered_titles else ""
else:
    query_input = st.text_input("Masukkan URL katalog...", placeholder="Contoh: https://openlibrary.telkomuniversity.ac.id/...")
    selected_title = ""
    if query_input:
        matches = df[df["url_katalog"].str.strip().str.lower() == query_input.strip().lower()]
        selected_title = matches.iloc[0]["judul"] if not matches.empty else ""

show_accuracy = st.checkbox("Tampilkan Akurasi (%)")

if st.button("🔎 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan masukkan dan pilih data yang tepat.")
    else:
        # Temukan index
        idx = df[df["judul"] == selected_title].index[0]
        recommendations = get_recommendations(idx)
        if not recommendations:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            st.success(f"Rekomendasi untuk: _{selected_title}_")
            col1, col2 = st.columns(2)
            for i, book in enumerate(recommendations):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(book["gambar"], width=150)
                    st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    if show_accuracy:
                        st.markdown(f"<div class='accuracy'>Akurasi: {book['akurasi']}%</div>", unsafe_allow_html=True)
                    st.markdown('</div>')
                    st.markdown("---")

st.markdown("## 📑 Rekomendasi Berdasarkan Kategori & Referensi Buku")

# ========== Dropdown Kategori ==========
col_a, col_b, col_c = st.columns(3)
with col_a:
    klasifikasi_options = [""] + sorted(df["klasifikasi"].dropna().unique().tolist())
    chosen_klasifikasi = st.selectbox("Klasifikasi:", klasifikasi_options)
with col_b:
    jenis_options = [""] + sorted(df["jenis"].dropna().unique().tolist())
    chosen_jenis = st.selectbox("Jenis:", jenis_options)
with col_c:
    subjek_options = [""] + sorted(df["subjek"].dropna().unique().tolist())
    chosen_subjek = st.selectbox("Subjek:", subjek_options)

# Filter berdasarkan kategori
filtered_df = df.copy()
if chosen_klasifikasi:
    filtered_df = filtered_df[filtered_df["klasifikasi"] == chosen_klasifikasi]
if chosen_jenis:
    filtered_df = filtered_df[filtered_df["jenis"] == chosen_jenis]
if chosen_subjek:
    filtered_df = filtered_df[filtered_df["subjek"] == chosen_subjek]

# Dropdown Referensi Buku
st.markdown("### Pilih Buku Referensi")
if not filtered_df.empty:
    reference_title = st.selectbox("Daftar Judul (sesuai kategori):", filtered_df["judul"].tolist())
else:
    st.info("Tidak ada buku di kategori ini.")
    reference_title = ""

show_accuracy_ref = st.checkbox("Tampilkan Akurasi (%) untuk Rekomendasi Kategori", key="show_acc_ref")

if st.button("🔗 Cari Rekomendasi Kategori"):
    if not reference_title:
        st.warning("⚠️ Silakan pilih Buku Referensi dari daftar.")
    else:
        idx_ref = df[df["judul"] == reference_title].index[0]
        # Termasuk referensi itu sendiri
        recommendations_ref = get_recommendations(idx_ref, top_n=5, include_self=True)
        if not recommendations_ref:
            st.error("❌ Tidak ada rekomendasi ditemukan di kategori ini.")
        else:
            st.success(f"Rekomendasi Kategori untuk: _{reference_title}_")
            col1r, col2r = st.columns(2)
            for i, book in enumerate(recommendations_ref):
                with (col1r if i % 2 == 0 else col2r):
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(book["gambar"], width=150)
                    st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    if show_accuracy_ref:
                        st.markdown(f"<div class='accuracy'>Akurasi: {book['akurasi']}%</div>", unsafe_allow_html=True)
                    st.markdown('</div>')
                    st.markdown("---")
