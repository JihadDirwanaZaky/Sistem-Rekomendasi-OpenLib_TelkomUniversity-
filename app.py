import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import gdown

# ========== Judul Aplikasi ==========
st.set_page_config(page_title="Sistem Rekomendasi Buku", layout="centered")
st.title("📚 Sistem Rekomendasi Buku & Jurnal")

# ========== Muat Data ==========
@st.cache_data
def load_data():
    # Pastikan ini adalah ID file yang benar, bukan folder
    file_id = "1Dejoa_9jrLf2MBC2UItQiggOf6rqIj2k"
    url = f"https://drive.google.com/uc?id= {file_id}"
    output = "katalog_bersih_pre-processed_ulang.csv"

    if not os.path.exists(output):
        try:
            st.info("⏳ Mengunduh file dari Google Drive...")
            gdown.download(url, output, quiet=False)
        except Exception as e:
            st.error(f"❌ Gagal mengunduh file: {e}")
            st.stop()

    try:
        df = pd.read_csv(output, nrows=50000)
    except Exception as e:
        st.error(f"❌ Error membaca file CSV: {e}")
        st.stop()

    required_columns = ["judul_clean", "combined_text", "url_katalog"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Kolom '{col}' tidak ditemukan dalam file CSV")
            st.stop()

    df.dropna(subset=required_columns, inplace=True)
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
        response = requests.get(url_katalog, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Sesuaikan dengan struktur HTML halaman katalog
        img_tag = soup.find('img', class_='thumbnail')  # Contoh: sesuaikan dengan kelas gambar sebenarnya
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
        else:
            return None
    except Exception as e:
        print(f"Error fetching image for {url_katalog}: {e}")
        return None

# ========== Cari Rekomendasi ==========
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

    filtered_scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in filtered_scores:
        results.append({
            "judul": df.iloc[i]["judul"],
            "url_katalog": df.iloc[i]["url_katalog"],
            "gambar": get_book_image(df.iloc[i]["url_katalog"]),
            "akurasi": round(score * 100, 2)
        })

    return results

# ========== UI Streamlit ==========
# ========== Input Judul + Dropdown Pilihan ==========
query_raw = st.text_input("Ketik sebagian judul buku...", placeholder="Contoh: Analisa")

filtered_titles = df[df["judul_clean"].str.contains(query_raw.strip(), case=False, na=False)]["judul"].unique().tolist()

if filtered_titles:
    selected_title = st.selectbox("Pilih judul lengkap:", filtered_titles)
else:
    selected_title = ""

show_accuracy = st.checkbox("Tampilkan Akurasi (%)")

# ========== Tombol Cari ==========
if st.button("🔍 Cari Rekomendasi"):
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
                    st.markdown(f"### {book['judul']}")
                    if book["gambar"]:
                        st.image(book["gambar"], width=150)
                    if show_accuracy:
                        st.markdown(f"**Akurasi**: {book['akurasi']}%")
                    st.markdown(f"[Lihat Detail]({book['url_katalog']})", unsafe_allow_html=True)
                    st.markdown("---")