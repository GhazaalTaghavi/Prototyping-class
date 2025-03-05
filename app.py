import streamlit as st
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load("model/classifier.pkl"), joblib.load("model/tfidf_vectorizer.pkl")

model, vectorizer = load_model()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("NewsCategorizer.csv")
        sample_articles = df.sample(5)[["headline", "short_description", "category"]]
    except Exception:
        df, sample_articles = None, None
    return df, sample_articles

df, sample_articles = load_data()

# Function to classify news
def classify_news(text):
    input_vector = vectorizer.transform([text])
    prediction = model.predict(input_vector)[0]
    
    sentiment = TextBlob(text).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

    return prediction, sentiment_label, sentiment

# Function to generate word cloud
def generate_word_cloud(category):
    category_words = " ".join(df[df["category"] == category]["short_description"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(category_words)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# Streamlit app layout
st.title("📰 News Categorization App")
st.subheader("Classify news articles into categories using NLP")
st.info("📌 Enter a news article manually, upload a text file, or select a sample article from the sidebar.")

# Sidebar with example selection
selected_article = ""
if sample_articles is not None:
    sample_selection = st.sidebar.selectbox("Select a sample news article:", ["None"] + sample_articles["headline"].tolist())
    if sample_selection != "None":
        selected_article = sample_articles[sample_articles["headline"] == sample_selection]["short_description"].values[0]

# UI Layout using columns
st.write("### Enter a news article for classification:")
text_area_height = st.slider("Adjust text area height", min_value=100, max_value=400, value=150)
news_text = st.text_area("", value=selected_article if selected_article else "", height=text_area_height)

# File Uploader Below Text Box
st.write("### Upload a File for Classification")
txt_file = st.file_uploader("Upload a text file", type=["txt"], label_visibility="collapsed")
if txt_file is not None:
    news_text = txt_file.read().decode("utf-8")

st.write("### Options")
sentiment_analysis = st.checkbox("Enable Sentiment Analysis", value=True)

if st.button("Classify News"):
    if news_text.strip():
        prediction, sentiment_label, sentiment = classify_news(news_text)
        
        st.success(f"Predicted Category: **{prediction}**")

        # Sentiment Analysis
        if sentiment_analysis:
            st.write(f"📝 Sentiment: {sentiment_label} ({sentiment:.2f})")
    else:
        st.warning("Please enter or select a news article!")

# Tabs for Word Cloud & Example Predictions
tab1, tab2 = st.tabs(["🌎 Word Cloud", "📖 Example Predictions"])

with tab1:
    st.write("### Word Cloud for Selected Category")
    category_choice = st.selectbox("Choose a category:", df["category"].unique(), key="wordcloud")
    st.pyplot(generate_word_cloud(category_choice))

with tab2:
    st.write("### See an Example of Each Category:")
    example_category = st.multiselect("Choose categories:", df["category"].unique(), default=[df["category"].unique()[0]])
    if example_category:
        for cat in example_category:
            st.write(f"**{cat} Example:**")
            example_text = df[df["category"] == cat]["short_description"].sample().values[0]
            st.text_area("", example_text, height=100, key=f"example_{cat}")

# Additional UI Enhancements
st.sidebar.markdown("---")
st.sidebar.markdown("🔍 *Developed as part of a Streamlit prototyping exercise* 📚")
