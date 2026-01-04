import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    layout="wide"
)
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This *Medical Q&A Chatbot* uses Natural Language Processing (NLP)
        techniques such as *TF-IDF* and *Cosine Similarity* to retrieve
        the most relevant medical answers from the *MedQuAD dataset*.
        """
    )
    st.markdown("---")
    st.write("*Built by:* Mohamed Sameer")
    st.write("*Domain:* NLP / Data Science")
    st.write("*Tech:* Python, Streamlit, Scikit-learn")

st.title("🩺 Medical Q&A Chatbot")

st.warning(
    "⚠️ This chatbot provides informational responses only and is not a substitute "
    "for professional medical advice."
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir,"medquad.csv")
    return pd.read_csv(data_path, encoding="utf-8")

dt = load_data()

# ================================
# PREPARE DATA
# ================================
corpus = dt["question"].astype(str)

@st.cache_data
def build_vectorizer(corpus):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(corpus)
    return tfidf, X

tfidf, X = build_vectorizer(corpus)

# ================================
# CHATBOT FUNCTION
# ================================
def medical_chatbot(user_question):
    user_vec = tfidf.transform([user_question.lower()])
    similarity = cosine_similarity(user_vec, X)

    score = similarity.max()
    index = similarity.argmax()

    if score < 0.2:
        return "❌ Sorry, I don't have information about this question."

    return dt.iloc[index]["answer"]

# ================================
# USER INPUT
# ================================
st.markdown("### Ask a medical question")
user_question = st.text_input(
    "",
    placeholder="e.g. What are the symptoms of glaucoma?"
)

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = medical_chatbot(user_question)
        st.success("Answer")
        st.markdown(
            f"""
            <div style="body
            background-color:#f9f9f9;
            padding:20px:
            border:1px solid #e0e0e0;
            line-height:1.6;
            font size:16px:
            ">
            {response}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption(
        "⚠️ Educational use only.This project is intended for learning and demonstration purpose.")
        

