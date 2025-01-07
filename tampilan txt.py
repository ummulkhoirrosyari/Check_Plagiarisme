import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time


# Fungsi untuk membaca teks dari file PDF, termasuk OCR jika perlu
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Jika teks ditemukan
                text += page_text + "\n"
            else:  # Jika halaman tidak memiliki teks, gunakan OCR
                st.write(f"Using OCR on page {page.page_number}...")
                page_image = page.to_image()
                ocr_text = pytesseract.image_to_string(page_image.original)
                text += ocr_text + "\n"
    return text


# Fungsi untuk scraping Google Scholar menggunakan Selenium
def scrape_google_scholar(query, max_results=5):
    # Setup Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Untuk menjalankan browser tanpa GUI
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Buka Google Scholar
    search_url = f"https://scholar.google.com/scholar?q={query}"
    driver.get(search_url)
    time.sleep(2)  # Tunggu halaman selesai dimuat

    # Ambil hasil pencarian
    links = []
    try:
        results = driver.find_elements(By.CSS_SELECTOR, 'h3.gs_rt a')  # Elemen hasil pencarian (judul dengan tautan)
        for result in results[:max_results]:  # Batasi hasil sesuai max_results
            links.append(result.get_attribute('href'))
    except Exception as e:
        st.write(f"Error during scraping: {e}")
    finally:
        driver.quit()  # Tutup browser

    return links


# Fungsi untuk validasi URL
def is_valid_url(url):
    regex = re.compile(
        r'^(https?://)'  # Harus dimulai dengan http:// atau https://
        r'(www\.)?'      # opsional www
        r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'  # domain
        r'([-a-zA-Z0-9()@:%_\+.~#?&/=]*)$'  # path
    )
    return re.match(regex, url) is not None


# Fungsi untuk mendeteksi plagiarisme
def detect_plagiarism(uploaded_text, sources):
    documents = [uploaded_text] + sources
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0][1:], vectorizer, tfidf_matrix


# Fungsi untuk membersihkan metadata dari teks
def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines
        if not any(keyword in line.lower() for keyword in ["issn", "volume", "nomor", "doi", "https", "http"])
    ]
    return " ".join(cleaned_lines[:3])  # Mengambil 3 baris pertama untuk query


# Fungsi untuk membersihkan metadata dari teks sumber
def clean_extracted_text(text):
    lines = text.splitlines()
    cleaned_lines = [
        line.strip() for line in lines
        if line.strip() and not any(keyword in line.lower() for keyword in ["font size", "help", "login", "register"])
    ]
    return " ".join(cleaned_lines)


# Aplikasi Streamlit
st.title("Check Plagiarism")

uploaded_file = st.file_uploader("Upload your document (PDF only):", type=["pdf"])
if uploaded_file:
    st.write("Extracting text...")
    uploaded_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text:", uploaded_text, height=200)

    # Memperbaiki ekstraksi referensi
    st.write("Analyzing References...")
    references = [line.strip() for line in uploaded_text.split("\n") if is_valid_url(line.strip())]
    if not references:
        st.write("No valid references found in the document.")
    else:
        for ref in references:
            st.write(f"Reference Found: {ref}")

    # Menambahkan fallback jika referensi kosong
    external_sources = [
        "https://example1.com/relevant-article",
        "https://example2.com/related-study"
    ]

    if not references:
        st.write("Using additional external sources...")
        references.extend(external_sources)

    # Membersihkan teks sebelum digunakan sebagai query
    cleaned_text = clean_text(uploaded_text)
    st.write("Searching for potential sources with the following query:")
    st.text(cleaned_text)

    # Mencari sumber dari Google Scholar menggunakan Selenium
    sources_links = scrape_google_scholar(cleaned_text, max_results=5)  # Maksimal 5 hasil
    all_sources_links = references + sources_links

    st.write("Found Sources:")
    for link in all_sources_links:
        st.write(f"[Source Link]({link})")

        # Mengambil teks dari sumber
        sources_texts = []
        for link in all_sources_links:
            try:
                page = requests.get(link)
                soup = BeautifulSoup(page.text, 'html.parser')
                sources_texts.append(clean_extracted_text(soup.get_text()))
            except Exception as e:
                st.write(f"Could not access {link}: {e}")
                continue

        st.write("Detecting plagiarism...")
        similarities, vectorizer, tfidf_matrix = detect_plagiarism(uploaded_text, sources_texts)
        avg_similarity = similarities.mean() if len(similarities) > 0 else 0

        st.write(f"Similarity Score: {avg_similarity * 100:.2f}%")

        # Menampilkan kalimat yang terdeteksi plagiarisme
        if len(similarities) > 0:
            st.write("Highlighted Plagiarized Sentences:")
            for i, similarity in enumerate(similarities):
                if similarity > 0.1:  # Threshold untuk plagiarisme
                    source_text = sources_texts[i]
                    st.write(f"From Source {i + 1}:")
                    st.text(" ".join(source_text.split()[:50]))  # Menampilkan 50 kata pertama

        # Membuat diagram pie
        labels = ['Original', 'Plagiarized']
        sizes = [100 - avg_similarity * 100, avg_similarity * 100]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

