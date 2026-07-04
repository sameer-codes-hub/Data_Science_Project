# 🏥 MedSQUAD Medical Chatbot

## Problem statement
Patients often struggle to find quick, reliable answers to medical questions. This NLP chatbot provides instant responses to medical queries using a large curated Q&A dataset.

## Results
- Trained on **16,000+ medical Q&A pairs** from the MedQuAD dataset
- Uses TF-IDF vectorisation and cosine similarity for answer retrieval
- Deployed as a conversational Streamlit web app

## Tech stack
- Python, NLTK, Scikit-learn
- NLP: TF-IDF, Cosine Similarity, Text Preprocessing
- Streamlit

## Live demo
[▶ Open Streamlit App](https://medquad-chat-bot.streamlit.app/)

## Run locally
```bash
git clone https://github.com/sameer-codes-hub/Data_Science_Project
cd Medical_Chatbot
pip install -r requirements.txt
streamlit run Chat_bot.py
```

## Files
| File | Description |
|------|-------------|
| `Chat_bot.py` | Main Streamlit app |
| `NLP_Chatbot.ipynb` | Model training notebook |
| `medquad.csv` | Medical Q&A dataset |



