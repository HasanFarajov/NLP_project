import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from docx import Document
from io import StringIO
from math import log

# İlk dəfə üçün lazımlı paketləri yüklə
nltk.download('punkt')
nltk.download('stopwords')

# === Xülasələmə funksiyası ===
def summarize_text(text, N=3):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)

    doc_per_word = {}
    for sentence in sentences:
        words = set(word_tokenize(sentence.lower()))
        words = [w for w in words if w.isalnum() and w not in stop_words]
        for word in words:
            doc_per_word[word] = doc_per_word.get(word, 0) + 1

    tf_matrix = []
    for sentence in sentences:
        words = [w for w in word_tokenize(sentence.lower()) if w.isalnum() and w not in stop_words]
        length = len(words)
        tf = {w: words.count(w)/length for w in set(words)} if length > 0 else {}
        tf_matrix.append(tf)

    idf_matrix = {w: log(total_sentences / (1 + doc_per_word[w])) for w in doc_per_word}

    tfidf_matrix = []
    for tf in tf_matrix:
        tfidf = {w: tf[w] * idf_matrix.get(w, 0) for w in tf}
        tfidf_matrix.append(tfidf)

    scores = [sum(tfidf.values()) / len(tfidf) if tfidf else 0 for tfidf in tfidf_matrix]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    summary = ' '.join([sentences[i] for i in sorted(top_indices)])
    return summary

# === Streamlit interfeys ===
st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("🧠 Text Summarization App")
st.markdown("Summarize long articles or documents using **TF-IDF**.")

option = st.radio("Choose Input Method:", ['Paste Text', 'Upload File'])

text = ""

if option == 'Paste Text':
    text = st.text_area("Paste your article or content below:", height=200)
elif option == 'Upload File':
    uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=['txt', 'docx'])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

if text:
    N = st.slider("Number of summary sentences", min_value=1, max_value=10, value=3)
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text, N)
            st.subheader("🔍 Summary:")
            st.success(summary)
else:
    st.info("Please paste text or upload a file to begin.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and NLTK")
