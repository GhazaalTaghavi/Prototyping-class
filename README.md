
## Project Overview
This project is an **interactive news categorization application** built with **Streamlit** and **Machine Learning (NLP)**.  
It allows users to classify news articles into predefined categories using a **Support Vector Classifier (SVC) with TF-IDF vectorization**.

The app provides additional insights such as:
- **News Classification**: Users can input a headline or short news text to determine its category.
- **Sample News Selection**: Users can test with sample news articles.
- **File Upload Support**: Users can upload a text file for classification.
- **Sentiment Analysis**: Determines whether a news article has a positive, neutral, or negative sentiment.
- **Word Cloud Visualization**: Shows the most frequent words in different news categories.

This prototype demonstrates **how AI-powered applications** can be built quickly using Streamlit and deployed for real-time classification.

---

## Features & Functionality
### **1. Model Choice**
- **Model Used:** Support Vector Classifier (SVC) with TF-IDF vectorization.
- **Course Background:** Cloud computing course from the first semester of MIBA in ESADE.
- **Why this model?**  
  SVC is computationally efficient, works well with TF-IDF text representation, and is lightweight compared to deep learning models.  
  It allows for **real-time predictions**, which is ideal for an interactive Streamlit app.

### **2. How the Prototype Works**
- **Users enter news text manually** or **upload a text file** for classification.
- **The app processes the text** using a trained **SVC model** and provides:
  - **Category prediction**
  - **Sentiment analysis (optional)**
- **Users can visualize** the most frequent words using a **word cloud**.

---

## Installation & Setup
### **1️. Install Dependencies**
Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### **2. Run the Streamlit App**
Execute the following command:

```bash
streamlit run app.py
```
---

## Challenges Faced
- **User Input Handling:** Managing seamless transitions between **manual text entry, file uploads, and  sample article selection**.
- **Performance Optimization:** Optimizing caching for **faster model loading** and **efficient dataset access**.
- **UI & Interactivity:** Creating a **clean and user-friendly Streamlit interface** with confidence scores, word clouds, and tabbed layouts.

---

## **Project Files**  
 **Main Application Files:**  
- `app.py` - The Streamlit application script.  

 **Model & Vectorizer:**  
- `model/classifier.pkl` - Trained classifier model.  
- `model/tfidf_vectorizer.pkl` - TF-IDF vectorizer used for text preprocessing.  

 **Dataset & Sample Data:**  
- `NewsCategorizer.csv` - Dataset used for training and testing the model.  

 **Requirements & Dependencies:**  
- `requirements.txt` - List of required dependencies for running the app.  

 **Documentation & Reports:**  
- `streamlit_project_summary.pdf` - A short document summarizing the project.  
- `README.md` - Project documentation (this file).  


