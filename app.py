import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import random
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ========== Konfigurasi Supabase ==========
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== Halaman Streamlit ==========
st.set_page_config(page_title="📚 Sistem Rekomendasi Buku", layout="wide")

st.markdown("""
<style>
    .book-card {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .book-card:hover {
        transform: scale(1.02);
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
</style>
""", unsafe_allow_html=True)

st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Load Data dari Supabase ==========
@st.cache_data(ttl=3600)
def load_data_from_supabase():
    try:
        response = client.table("katalog_buku").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"❌ Error saat mengambil data dari Supabase: {e}")
        st.stop()

df = load_data_from_supabase()

# Pastikan kolom penting tersedia
required_columns = ["judul", "judul_clean", "url_katalog", "klasifikasi_clean", "jenis", "subjek_clean", "combined_text"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan dalam tabel 'katalog_buku'")
        st.stop()

# ========== Setup TF-IDF dan Cosine Similarity ==========
@st.cache_resource
def setup_model():
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, cosine_sim

vectorizer, tfidf_matrix, cosine_sim = setup_model()

# ========== Ambil Gambar dari URL ==========
def get_book_image(url_katalog):
    try:
        response = requests.get(url_katalog, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', class_='thumbnail') or soup.find('img', class_='cover')
        return img_tag['src'] if img_tag and 'src' in img_tag.attrs else "https://via.placeholder.com/150x220?text=No+Image"
    except Exception as e:
        print(f"Error fetching image for {url_katalog}: {e}")
        return "https://via.placeholder.com/150x220?text=No+Image"

# ========== Fungsi Rekomendasi Berdasarkan Judul ========== 
def get_recommendations(query, top_n=5):
    query = query.strip().lower()
    if not query:
        return []
    matches = df[df["judul_clean"].str.contains(query, case=False, na=False)]
    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in top_scores:
        results.append({
            "judul": df.iloc[i]["judul"],
            "url": df.iloc[i]["url_katalog"],
            "gambar": get_book_image(df.iloc[i]["url_katalog"]),
            "akurasi": round(score * 100, 2)
        })

    return results

# ========== Fungsi Rekomendasi Berdasarkan Filter ==========
def get_by_filter(kls, jenis, subjek, ref_judul):
    filtered = df[
        (df["klasifikasi_clean"] == kls) &
        (df["jenis"] == jenis) &
        (df["subjek_clean"] == subjek)
    ]
    if filtered.empty:
        return []

    referensi = df[df["judul"] == ref_judul]
    if referensi.empty:
        return []

    idx_ref = referensi.index[0]
    sim_scores = list(enumerate(cosine_sim[idx_ref][filtered.index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:5]]

    return [{
        "judul": df.iloc[i]["judul"],
        "url": df.iloc[i]["url_katalog"],
        "gambar": get_book_image(df.iloc[i]["url_katalog"]),
        "akurasi": round(sim_scores[j][1] * 100, 2)
    } for j, i in enumerate(top_indices)]

# ========== Carousel 5 Buku Acak ==========
st.subheader("🎯 Hari Ini di Katalog")
cols = st.columns(5)
random.seed(int(time.time()) % 60)
sample_indices = random.sample(range(len(df)), min(5, len(df)))

for col, idx in zip(cols, sample_indices):
    book = df.iloc[idx]
    with col:
        st.image(get_book_image(book['url_katalog']), width=120)
        st.markdown(f"[{book['judul']}]({book['url_katalog']})", unsafe_allow_html=True)

# Auto-refresh setiap 15 detik
st_autorefresh(interval=15000, limit=None, key="refresh")

st.markdown("---")

# ========== Cari Rekomendasi Berdasarkan Judul atau URL ==========
st.subheader("🔍 Cari Rekomendasi")

search_type = st.radio("Cari berdasarkan:", ["Judul", "URL"], horizontal=True)

query = st.text_input("Masukkan kata kunci...")

if search_type == "Judul":
    filtered_options = df[df["judul_clean"].str.contains(query.strip(), case=False, na=False)]["judul"].unique()
else:
    filtered_options = df[df["url_katalog"].str.contains(query.strip(), case=False, na=False)]["url_katalog"].unique()

selected = st.selectbox("Pilih dari daftar:", filtered_options) if len(filtered_options) > 0 else ""

if st.button("🔎 Cari Rekomendasi"):
    if not selected:
        st.warning("⚠️ Silakan pilih judul/url dari daftar.")
    else:
        if search_type == "Judul":
            hasil = get_recommendations(selected)
        else:
            hasil = get_recommendations(df[df["url_katalog"] == selected]["judul"].iloc[0])

        if not hasil:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            st.success(f"Rekomendasi untuk: _{selected}_")
            for item in hasil:
                st.markdown('<div class="book-card">', unsafe_allow_html=True)
                st.image(item["gambar"], width=130)
                st.markdown(f"<a class='book-title' href='{item['url']}'>{item['judul']}</a>", unsafe_allow_html=True)
                st.markdown(f"<div class='accuracy'>Akurasi: {item['akurasi']}%</div>", unsafe_allow_html=True)
                st.markdown("</div>")
                st.markdown("---")

st.markdown("---")

# ========== Rekomendasi Berdasarkan Kombinasi Klasifikasi ==========
st.subheader("🧠 Rekomendasi Berdasarkan Kategori & Referensi")

col1, col2, col3, col4 = st.columns(4)

with col1:
    klasifikasi = st.selectbox("Klasifikasi", options=sorted(df["klasifikasi_clean"].dropna().unique()))

with col2:
    jenis = st.selectbox("Jenis", options=sorted(df["jenis"].dropna().unique()))

with col3:
    subjek = st.selectbox("Subjek", options=sorted(df["subjek_clean"].dropna().unique()))

with col4:
    ref_judul = st.selectbox("Buku Referensi", options=df["judul"].unique())

if st.button("Tampilkan Rekomendasi Berdasarkan Kategori"):
    hasil = get_by_filter(klasifikasi, jenis, subjek, ref_judul)

    if not hasil:
        st.error("❌ Tidak ada rekomendasi ditemukan.")
    else:
        st.success(f"Rekomendasi untuk kategori {klasifikasi} - {jenis} - {subjek}")
        for item in hasil:
            st.markdown('<div class="book-card">', unsafe_allow_html=True)
            st.image(item["gambar"], width=130)
            st.markdown(f"<a class='book-title' href='{item['url']}'>{item['judul']}</a>", unsafe_allow_html=True)
            st.markdown(f"<div class='accuracy'>Akurasi: {item['akurasi']}%</div>", unsafe_allow_html=True)
            st.markdown("</div>")
            st.markdown("---")
