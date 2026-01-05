🩺 Medical Q&A Chatbot Using NLP (TF-IDF)
🚀 Live Demo (Streamlit App)

👉 https://medquad-chat-bot.streamlit.app/

📌 Project Overview

This project is a Medical Question & Answer Chatbot built using Natural Language Processing (NLP) techniques.
It allows users to ask medical-related questions and returns the most relevant answer from the MedQuAD dataset using TF-IDF vectorization and cosine similarity.

⚠️ This chatbot is for educational and informational purposes only and is not a substitute for professional medical advice.

🚀 Features

NLP-based question answering system

TF-IDF vectorization for text representation

Cosine similarity for best answer matching

Uses real MedQuAD medical dataset

Interactive UI using Streamlit

Fully deployed on Streamlit Cloud

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Streamlit

NLP (TF-IDF, Cosine Similarity)

📂 Project Files

Chat_bot.py – Streamlit application (main file)

medquad.csv – Medical Q&A dataset

NLP_Chatbot.ipynb – Model experimentation notebook

NLP_PREPROCESS.ipynb – Text preprocessing steps

requirements.txt – Required Python libraries

runtime.txt – Python version configuration

▶️ How to Run This Project Locally

Install required packages

pip install -r requirements.txt


Run the Streamlit app

streamlit run Chat_bot.py

🎯 Output

User enters a medical question

Chatbot finds the most similar medical question

Displays the best-matched medical answer

Handles unknown questions gracefully

🧠 How It Works (NLP Pipeline)

Text cleaning & preprocessing

TF-IDF vectorization of questions

Cosine similarity calculation

Best answer retrieval based on similarity score

Threshold-based fallback response

📊 Example Use Cases

“What are the symptoms of glaucoma?”

“How is diabetes diagnosed?”

“What causes high blood pressure?”

👤 Author

Mohamed Sameer
📌 Domain: Data Science | NLP | Machine Learning
📌 Tools: Python, Streamlit, Scikit-learn

⭐ Note

If you find this project useful, feel free to star the repository ⭐
