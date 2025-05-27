import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import random
from supabase import create_client, Client

# === Konfigurasi Supabase ===
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
</style>
""", unsafe_allow_html=True)

st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Muat Data dari Supabase ==========
@st.cache_data
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
required_columns = ["judul_clean", "subjek_clean", "url_katalog"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan di database Supabase")
        st.stop()

# Buat combined_text dari 2 kolom
df["combined_text"] = df["judul_clean"].fillna("") + " " + df["subjek_clean"].fillna("")
df.reset_index(drop=True, inplace=True)

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
        response = requests.get(url_katalog, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        img_tag = soup.find('img', class_='thumbnail') or soup.find('img', class_='cover')
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
        else:
            return "https://via.placeholder.com/150x220?text=No+Image "
    except Exception as e:
        print(f"Error fetching image for {url_katalog}: {e}")
        return "https://via.placeholder.com/150x220?text=No+Image "

# ========== Fungsi Rekomendasi Berbasis Dua Kolom ==========
def get_recommendations(query, top_n=5):
    query = query.strip().lower()
    if not query:
        return []

    # Cari judul atau subjek yang cocok
    matches_judul = df[df["judul_clean"].str.contains(query, case=False, na=False)]
    matches_subjek = df[df["subjek_clean"].str.contains(query, case=False, na=False)]

    # Gabung hasil
    matches = pd.concat([matches_judul, matches_subjek]).drop_duplicates(subset=["id"])

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Hilangkan dokumen yang sama
    filtered_scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in filtered_scores:
        results.append({
            "judul": df.iloc[i]["judul"],
            "url_katalog": df.iloc[i]["url_katalog"],
            "gambar": get_book_image(df.iloc[i]["url_katalog"]),
            "akurasi": round(score[1] * 100, 2)
        })

    return results

# ========== Today's Catalog Preview (5 Buku Acak) ==========
st.header("📖 Hari Ini di Katalog Buku")
cols_today = st.columns(5)

random_indices = random.sample(range(len(df)), min(5, len(df)))

for col, i in zip(cols_today, random_indices):
    with col:
        row = df.iloc[i]
        img_url = get_book_image(row["url_katalog"])
        st.image(img_url, width=130)
        st.markdown(f"[{row['judul']}]({row['url_katalog']})", unsafe_allow_html=True)

st.markdown("---")

# ========== UI Streamlit ==========
query_raw = st.text_input("🔍 Ketik sebagian judul buku...", placeholder="Contoh: Analisis")

# Gunakan autocomplete alih-alih dropdown
filtered_titles = df["judul"].unique().tolist()
selected_title = st_autocomplete(
    key="autocomplete_search",
    label="Pilih judul buku:",
    options=filtered_titles,
    placeholder="Masukkan judul buku...",
    clearable=True,
    max_options=10  # Batas jumlah opsi yang ditampilkan
)

show_accuracy = st.checkbox("Tampilkan Akurasi (%)")

# ========== Tombol Cari ==========
if st.button("🔎 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan pilih judul dari dropdown.")
    else:
        with st.spinner("Memuat hasil..."):
            recommendations = get_recommendations(selected_title)

        if not recommendations:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            st.success(f"Rekomendasi untuk: _{selected_title}_")
            cols = st.columns(2)

            for i, book in enumerate(recommendations):
                with cols[i % 2]:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(book["gambar"], width=150)
                    st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    if show_accuracy:
                        st.markdown(f"<div class='accuracy'>Akurasi: {book['akurasi']}%</div>", unsafe_allow_html=True)
                    st.markdown('</div>')
                    st.markdown("---")
