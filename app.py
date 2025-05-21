import streamlit as st
import nltk
import os
import ssl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from docx import Document
from io import StringIO
from math import log

# Patch SSL and download nltk resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Summarization function
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
    tfidf_matrix = [{w: tf[w] * idf_matrix.get(w, 0) for w in tf} for tf in tf_matrix]
    scores = [sum(tfidf.values()) / len(tfidf) if tfidf else 0 for tfidf in tfidf_matrix]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    summary = ' '.join([sentences[i] for i in sorted(top_indices)])
    return summary

# Streamlit UI
st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("🧠 Text Summarizer")
st.markdown("Summarize long articles or documents using TF-IDF.")

option = st.radio("Select input method:", ['Paste Text', 'Upload File'])

text = ""

if option == 'Paste Text':
    text = st.text_area("Paste your content here:", height=200)
elif option == 'Upload File':
    uploaded_file = st.file_uploader("Upload a `.txt` or `.docx` file", type=['txt', 'docx'])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            except Exception as e:
                st.error("⚠️ Failed to read .docx file. Make sure the file is not corrupted.")
                st.stop()


if text:
    N = st.slider("Number of summary sentences", min_value=1, max_value=10, value=3)
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text, N)
            st.subheader("🔍 Summary")
            st.success(summary)
else:
    st.info("Please provide text input or upload a file to begin.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and NLTK")
