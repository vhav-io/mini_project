# mini_project
# 📧 Email/SMS Spam Detector

### A Machine Learning Mini-Project
**Course:** B.Tech Computer Science (Data Science) (3rd Sem)  
**Subject:** Mini Project
---

## 📖 About The Project
This project is a Machine Learning application that classifies text messages (SMS or Emails) as either **"Spam"** (Unwanted/Fake) or **"Not Spam"** (Ham/Legitimate). 

It uses **Natural Language Processing (NLP)** to analyze the text and the **Multinomial Naive Bayes** algorithm to predict the category. The project features a user-friendly web interface built with **Streamlit**.

## 🚀 Features
- **Instant Classification:** Real-time prediction of messages.
- **Visual Feedback:** Green alerts for safe messages, Red alerts for spam.
- **Web Interface:** Simple and clean UI using Streamlit.
- **Lightweight:** Runs easily on local machines or cloud environments like GitHub Codespaces.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **GUI Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (MultinomialNB)
* **Data Processing:** Pandas, NumPy
* **Text Processing:** CountVectorizer (Bag of Words)

---

## ⚙️ How to Run Locally (or in Codespaces)

### 1. Prerequisites
Make sure you have Python installed. If you are in GitHub Codespaces, Python is already installed.

### 2. Install Dependencies
Run the following command in your terminal to install the required libraries:
```bash
pip install pandas scikit-learn streamlit