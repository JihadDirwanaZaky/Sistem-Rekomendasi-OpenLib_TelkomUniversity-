import pandas as pd
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client

# ========== Konfigurasi Supabase ==========
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== Halaman Streamlit ==========
st.set_page_config(page_title="📚 Sistem Rekomendasi Buku & Jurnal", layout="wide")

st.title("📚 Sistem Rekomendasi Buku & Jurnal Open Library Telkom University")

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

# ========== Form Pencarian ==========
st.subheader("🔍 Cari Rekomendasi")

search_type = st.radio("Cari berdasarkan:", ["Judul", "URL"], horizontal=True, key="search_type")
query = st.text_input("Masukkan kata kunci...", key="query_input")

# Inisialisasi session state
if "filtered_options" not in st.session_state:
    st.session_state.filtered_options = []
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

# Tombol 1: Cari Judul/URL → Memperbarui dropdown
if st.button("🔍 Cari Judul"):
    if not query:
        st.warning("⚠️ Silakan masukkan kata kunci.")
    else:
        if search_type == "Judul":
            st.session_state.filtered_options = df[df["judul_clean"].str.contains(query.strip(), case=False, na=False)]["judul"].unique().tolist()
        else:
            st.session_state.filtered_options = df[df["url_katalog"].str.contains(query.strip(), case=False, na=False)]["url_katalog"].unique().tolist()

# Dropdown pilihan (hanya muncul jika ada hasil)
if st.session_state.filtered_options:
    st.session_state.selected_item = st.selectbox("Pilih dari hasil pencarian:", st.session_state.filtered_options)

# Tombol 2: Cari Rekomendasi
if st.button("📚 Tampilkan Rekomendasi"):
    if not st.session_state.selected_item:
        st.warning("⚠️ Silakan pilih judul atau URL terlebih dahulu.")
    else:
        if search_type == "Judul":
            hasil = get_recommendations(st.session_state.selected_item)
        else:
            match = df[df["url_katalog"] == st.session_state.selected_item]
            if match.empty:
                st.error("❌ URL tidak ditemukan dalam database.")
                st.stop()
            hasil = get_recommendations(match["judul"].iloc[0])

        if not hasil:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            st.success(f"Rekomendasi untuk: _{st.session_state.selected_item}_")
            cols = st.columns(2)

            for i, item in enumerate(hasil):
                with cols[i % 2]:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(item["gambar"], width=130)
                    st.markdown(f"<a class='book-title' href='{item['url']}' target='_blank'>{item['judul']}</a>", unsafe_allow_html=True)
                    st.markdown(f"<span class='accuracy'>Akurasi: {item['akurasi']}%</span>", unsafe_allow_html=True)
                    st.markdown("</div>")
                    st.markdown("---")
