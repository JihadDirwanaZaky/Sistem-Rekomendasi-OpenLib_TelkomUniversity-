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
required_columns = ["judul_clean", "url_katalog", "combined_text"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan di database Supabase")
        st.stop()

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

# ========== Fungsi Rekomendasi Berbasis Satu Kolom (combined_text) ==========
def get_recommendations(query, top_n=5):
    query = query.strip().lower()
    if not query:
        return []

    # Cari judul yang cocok berdasarkan combined_text
    matches = df[df["combined_text"].str.contains(query, case=False, na=False)]
    
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

# === CSS Styling ===
st.markdown("""
<style>
    .autocomplete-container {
        position: relative;
    }
    .autocomplete-input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 16px;
    }
    .autocomplete-results {
        position: absolute;
        z-index: 1;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ccc;
        background-color: white;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .autocomplete-result {
        padding: 10px;
        cursor: pointer;
    }
    .autocomplete-result:hover {
        background-color: #f1f1f1;
    }
</style>
""", unsafe_allow_html=True)

# === Autocomplete Search Box ===
st.markdown("""
<div class="autocomplete-container">
    <input type="text" id="autocomplete-input" placeholder="🔍 Ketik sebagian judul buku..." />
    <div id="autocomplete-results" class="autocomplete-results"></div>
</div>
<script>
    const judulList = JSON.parse(`""" + str(df["judul_clean"].tolist()) + """`);
    const inputElement = document.getElementById("autocomplete-input");
    const resultsContainer = document.getElementById("autocomplete-results");

    inputElement.addEventListener("input", function() {
        const query = this.value.toLowerCase();
        resultsContainer.innerHTML = "";

        if (!query) {
            return;
        }

        const suggestions = judulList.filter(judul => judul.toLowerCase().includes(query));
        suggestions.forEach(judul => {
            const resultDiv = document.createElement("div");
            resultDiv.className = "autocomplete-result";
            resultDiv.textContent = judul;
            resultDiv.addEventListener("click", () => {
                inputElement.value = judul;
                resultsContainer.innerHTML = "";
            });
            resultsContainer.appendChild(resultDiv);
        });
    });
</script>
""", unsafe_allow_html=True)

# === Ambil Input dari Autocomplete ===
selected_title = st.session_state.get("selected_title", "")

show_accuracy = st.checkbox("Tampilkan Akurasi (%)")

# === Tombol Cari ===
if st.button("🔎 Cari Rekomendasi"):
    if not selected_title:
        st.warning("⚠️ Silakan pilih judul dari daftar.")
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
