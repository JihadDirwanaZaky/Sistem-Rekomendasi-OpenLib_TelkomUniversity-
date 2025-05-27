import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from supabase import create_client, Client

# ========== Konfigurasi Supabase ==========
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

# Pastikan kolom penting tersedia
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

# ========== Fungsi Rekomendasi Berbasis URL ==========
def get_recommendations_by_url(url_katalog, top_n=5):
    matches = df[df["url_katalog"] == url_katalog]
    if matches.empty:
        return []
    
    idx = matches.index[0]  # Ambil index pertama
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

# ========== Parsing URL ==========
query_params = st.query_params
raw_url = query_params.get("url", "")

if raw_url:
    parsed_url = raw_url.strip().lower()
    recommendations = get_recommendations_by_url(parsed_url)

else:
    st.info("🔗 Silakan gunakan format berikut:")
    st.code("https://openlibraryrecommend.streamlit.app/?url=https ://openlibrary.telkomuniversity.ac.id/home/catalog/id/232959/slug/...", language="text")
    recommendations = []

# ========== UI Streamlit ==========
if recommendations:
    st.success(f"Rekomendasi untuk buku dari URL: _{parsed_url}_")
    cols = st.columns(2)

    for i, book in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown('<div class="book-card">', unsafe_allow_html=True)
            st.image(book["gambar"], width=150)
            st.markdown(f"<a class='book-title' href='{book['url_katalog']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
            st.markdown(f"<span class='accuracy-tag'>{book['akurasi']}%</span>", unsafe_allow_html=True)
            st.markdown('</div>')
            st.markdown("---")

# ========== Manual Search ==========
else:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("🔍 Atau cari manual menggunakan judul atau URL:")

    query = st.text_input("Ketik sebagian judul atau URL...")

    if query:
        # Cari judul
        filtered_titles = df[df["judul"].str.contains(query, case=False, na=False)]["judul"].unique().tolist()
        # Cari URL
        filtered_urls = df[df["url_katalog"].str.contains(query, case=False, na=False)]["url_katalog"].unique().tolist()
        
        if filtered_urls:
            selected_url = st.selectbox("Pilih URL:", options=filtered_urls)
            recommendations = get_recommendations_by_url(selected_url)
            
            if recommendations:
                st.success(f"Rekomendasi untuk: _{selected_url}_")
                cols = st.columns(2)
                for i, book in enumerate(recommendations):
                    with cols[i % 2]:
                        st.markdown('<div class="book-card">', unsafe_allow_html=True)
                        st.image(book["gambar"], width=150)
                        st.markdown(f"[{book['judul']}]({book['url_katalog']})", unsafe_allow_html=True)
                        st.markdown(f"<span class='accuracy-tag'>{book['akurasi']}%</span>", unsafe_allow_html=True)
                        st.markdown('</div>')
                        st.markdown("---")
            else:
                st.warning("⚠️ Tidak ada rekomendasi ditemukan.")
        elif filtered_titles:
            selected_title = st.selectbox("Pilih judul:", options=filtered_titles[:50])
            idx = df[df["judul"] == selected_title].index[0]
            recommendations = get_recommendations_by_url(df.loc[idx, "url_katalog"])
        else:
            st.info("Tidak ada judul atau URL yang cocok.")

# ========== Footer ==========
st.markdown("<br><center>© Telkom University Library Recommendation System</center>", unsafe_allow_html=True)
