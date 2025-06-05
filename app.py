import streamlit as st
import pandas as pd
import requests
import random
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ========= Supabase Config =========
SUPABASE_URL = "https://vmmzsghhyrtddnsmoscw.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtbXpzZ2hoeXJ0ZGRuc21vc2N3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzYyNTYsImV4cCI6MjA2MzQ1MjI1Nn0.V6G6FTo5hSjYtmGzoHiJz1ez_tcFDhpwkn9qyQlFa0Q"

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========= Streamlit Page Config =========
st.set_page_config(page_title="📚 Sistem Rekomendasi Buku", layout="wide")

# ========= Custom CSS for white background and clear text =========
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #ffffff !important;
            color: #2c3e50 !important;
        }
        .book-title {
            font-weight: bold;
            font-size: 1rem;
            color: #1a237e;
        }
        .accuracy {
            font-size: 0.9rem;
            color: #555;
        }
        .recommend-button {
            background-color: #e8f0fe;
            color: #1a73e8;
            border-radius: 4px;
            padding: 5px 8px;
            margin-top: 5px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)


st.title("📚 Sistem Rekomendasi Buku & Jurnal Open Library Telkom University")

# ========= Load Dataset =========
@st.cache_data(ttl=3600)
def load_data():
    data = client.table("katalog_buku").select("*").execute()
    df = pd.DataFrame(data.data)
    return df

df = load_data()

# Pastikan kolom penting ada
required_columns = ["judul", "judul_clean", "url_katalog", "combined_text"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan.")
        st.stop()

# ========= TF-IDF Model =========
@st.cache_resource
def build_model():
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, cosine_sim

vectorizer, tfidf_matrix, cosine_sim = build_model()

# ========= Ambil Gambar =========
def get_book_image(url_katalog):
    try:
        response = requests.get(url_katalog, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        img = soup.find('img', class_='thumbnail') or soup.find('img', class_='cover')
        return img['src'] if img and 'src' in img.attrs else "https://via.placeholder.com/150x220?text=No+Image"
    except:
        return "https://via.placeholder.com/150x220?text=No+Image"

# ========= Rekomendasi Berdasarkan Judul =========
def get_recommendations_by_title(title, top_n=5):
    match = df[df["judul"] == title]
    if match.empty:
        return []

    idx = match.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    results = []
    for i, score in top_scores:
        row = df.iloc[i]
        results.append({
            "judul": row["judul"],
            "url": row["url_katalog"],
            "gambar": get_book_image(row["url_katalog"]),
            "akurasi": round(score * 100, 2)
        })

    return results

# ========= Random Static Catalog Preview =========
if "preview_books" not in st.session_state:
    st.session_state.preview_books = df.sample(5, random_state=42)

st.subheader("🎯 Try These Catalogs")
cols = st.columns(5)
for col, (_, book) in zip(cols, st.session_state.preview_books.iterrows()):
    with col:
        st.image(get_book_image(book["url_katalog"]), width=110)
        st.markdown(f"<a class='book-title' href='{book['url_katalog']}'>{book['judul']}</a>", unsafe_allow_html=True)

st.markdown("---")

# ========= Pencarian dan Rekomendasi =========
if "judul_input" not in st.session_state:
    st.session_state.judul_input = ""

st.subheader("🔍 Cari Rekomendasi Berdasarkan Judul")

judul_input = st.text_input("Ketik sebagian judul:", value=st.session_state.judul_input, key="judul_text")
filtered_judul_list = df[df["judul_clean"].str.contains(judul_input.strip().lower(), na=False)]["judul"].unique().tolist()

selected_title = None
if filtered_judul_list:
    selected_title = st.selectbox("Pilih Judul:", filtered_judul_list)

if st.button("📚 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan pilih judul dari dropdown.")
    else:
        rekomendasi = get_recommendations_by_title(selected_title)

        if not rekomendasi:
            st.error("❌ Tidak ada rekomendasi ditemukan.")
        else:
            st.success(f"Rekomendasi untuk: _{selected_title}_")
            cols = st.columns(2)

            for i, book in enumerate(rekomendasi):
                with cols[i % 2]:
                    st.image(book["gambar"], width=130)
                    st.markdown(f"<a class='book-title' href='{book['url']}' target='_blank'>{book['judul']}</a>", unsafe_allow_html=True)
                    st.markdown(f"<div class='accuracy'>Akurasi: {book['akurasi']}%</div>", unsafe_allow_html=True)

                    # Tombol Recommend by this
                    if st.button("📎 Recommend by this", key=f"recom_btn_{i}"):
                        st.session_state.judul_input = book["judul"]
                        st.experimental_rerun()
