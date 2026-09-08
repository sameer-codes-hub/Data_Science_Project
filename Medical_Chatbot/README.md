# 🩺 MedSQUAD – Medical Q&A Retrieval Chatbot

MedSQUAD is an NLP-based medical question-answering retrieval chatbot that retrieves relevant answers to healthcare-related questions from a medical Q&A dataset.

The project demonstrates practical applications of Natural Language Processing (NLP), text representation, TF-IDF vectorization, cosine similarity, and Streamlit deployment.

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://medquad-chat-bot.streamlit.app/

> Note: The Streamlit application may go to sleep after periods of inactivity. If prompted, click **"Yes, get this app back up!"** to wake the application.

---

## 📌 Project Overview

The objective of this project is to build an interactive system that can identify the most relevant answer to a user's medical-related question from a collection of existing medical Q&A pairs.

Instead of generating new medical responses, MedSQUAD uses an information-retrieval approach to find the most relevant existing answer based on textual similarity.

---

## 🧠 How It Works

The chatbot follows an NLP-based retrieval pipeline:

1. User enters a healthcare-related question.
2. The input text is converted into a TF-IDF representation..
3. The medical Q&A dataset is converted into numerical representations using **TF-IDF**.
4. **Cosine similarity** is calculated between the user's question and available questions.
5. The most relevant matching Q&A pair is identified.
6. The corresponding answer is displayed through the Streamlit interface.

### NLP Pipeline

```text

User Question
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Most Relevant Q&A Pair
      ↓
Answer Retrieval
      ↓
Streamlit Interface

```
## 📊 Dataset

The project uses the MedQuAD dataset, containing more than 16,000 medical question-answer pairs.

The dataset provides the underlying collection of questions and answers used by the retrieval system.

The system retrieves relevant answers from the dataset rather than generating medical diagnoses or treatment recommendations.

## 🔧 NLP Techniques

The project uses the following NLP and information-retrieval techniques:

- TF-IDF Vectorization
- Cosine Similarity
- Text Matching
- Information Retrieval

These techniques are used to transform and compare text so that the system can identify relevant answers.

## 🛠️ Tech Stack

### Programming Language

- Python

### Libraries & Frameworks

- Pandas
- Scikit-learn
- Streamlit
  
 ### NLP
 
- Natural Language Processing
- TF-IDF
- Cosine Similarity
- Information Retrieval

## 📁 Project Structure

```text
Data_Science_Project/
│
├── Medical_Chatbot/
│   ├── Chat_bot.py
│   └── README.md
│
├── Chat_bot.py
├── medquad.csv
├── requirements.txt
└── other project files
```

## 💻 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sameer-codes-hub/Data_Science_Project.git
```

### 2. Navigate to the project folder

```bash
cd Data_Science_Project
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run Chat_bot.py
```

The application will open in your local browser.

## 🎯 Key Learning Outcomes

Through this project, I strengthened my practical understanding of:

- Natural Language Processing
- Information retrieval
- TF-IDF vectorization
- Cosine similarity
- Text matching
- Python data processing
- Scikit-learn
- Streamlit application development
- Deploying machine-learning applications

## ⚠️ Medical Disclaimer

MedSQUAD is an educational and informational NLP project.

It is not a medical professional and should not be used for medical diagnosis, treatment decisions, or emergency situations.

Always consult a qualified healthcare professional for medical advice.

## 👨‍💻 Author

**Mohamed Sameer Hamad**

Data Analyst | Data Science & Machine Learning | AI & GenAI

🔗 **GitHub:**
https://github.com/sameer-codes-hub

🔗 **LinkedIn:**
https://www.linkedin.com/in/mohamed-sameer-hamad-585489292/



