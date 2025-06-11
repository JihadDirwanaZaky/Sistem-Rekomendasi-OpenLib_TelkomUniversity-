import streamlit as st
import pandas as pd
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
    html, body, .main {
        background-color: white !important;
        color: #2c3e50 !important;
        font-family: 'Segoe UI', sans-serif;
    }
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
        color: #1a237e;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Sistem Rekomendasi Buku & Jurnal Open Library Telkom University")

# ========== Load Data dari Supabase ==========
@st.cache_data(ttl=3600)
def load_data_from_supabase():
    try:
        response = client.table("katalog_buku").select("*").execute()
        df = pd.DataFrame(response.data)

        # Bersihkan kolom judul dari tag HTML
        df["judul"] = df["judul"].apply(lambda x: BeautifulSoup(str(x), "html.parser").get_text(strip=True) if pd.notna(x) else "")
        
        return df
    except Exception as e:
        st.error(f"❌ Error saat mengambil data dari Supabase: {e}")
        st.stop()

df = load_data_from_supabase()

# Pastikan kolom penting tersedia
required_columns = ["judul", "judul_clean", "url_katalog", "combined_text"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan dalam tabel 'katalog_buku'")
        st.stop()

# ========== Setup TF-IDF dan Cosine Similarity ==========
@st.cache_resource
def setup_model():
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
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
    except:
        return "https://via.placeholder.com/150x220?text=No+Image"

# ========== Fungsi Rekomendasi Berdasarkan Judul ========== 
def get_recommendations_by_title(title, top_n=5):
    match = df[df["judul"] == title]
    if match.empty:
        return []

    idx = match.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in top_scores:
        row = df.iloc[i]
        results.append({
            "judul": row["judul"],
            "url": row["url_katalog"],
            "gambar": get_book_image(row["url_katalog"])
        })

    return results

# ========== Carousel 5 Buku Statis ==========
if "preview_books" not in st.session_state:
    st.session_state.preview_books = df.sample(5, random_state=42)

st.subheader("🎯 Hari Ini di Katalog")
cols = st.columns(5)

for col, (_, book) in zip(cols, st.session_state.preview_books.iterrows()):
    with col:
        st.image(get_book_image(book["url_katalog"]), width=110)
        st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)

st.markdown("---")

# ========== Cari Rekomendasi Berdasarkan Judul ==========
st.subheader("🔍 Cari Rekomendasi Berdasarkan Judul")

# Input pencarian
judul_input = st.text_input("Ketik sebagian judul buku...")

# Filter judul berdasarkan input
filtered_judul_list = df[df["judul_clean"].str.contains(judul_input.strip().lower(), na=False)]["judul"].unique().tolist()

# Dropdown pilihan judul
selected_title = ""
if filtered_judul_list:
    selected_title = st.selectbox("Pilih judul:", filtered_judul_list)
else:
    st.info("⚠️ Tidak ada judul yang cocok dengan kata kunci.")

# Tombol cari rekomendasi
if st.button("🔎 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan pilih judul dari dropdown.")
    else:
        hasil_rekomendasi = get_recommendations_by_title(selected_title)

        if not hasil_rekomendasi:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            cols = st.columns(2)
            for i, book in enumerate(hasil_rekomendasi):
                with cols[i % 2]:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(book["gambar"], width=130)
                    st.markdown(f"<a class='book-title' href='{book['url']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    st.markdown("</div>")
                    st.markdown("---")
