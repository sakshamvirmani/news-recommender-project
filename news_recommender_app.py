import streamlit as st
import streamlit.components.v1 as components
import requests
import spacy
from functools import lru_cache
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.cloud import language_v1

st.write("✅ App has started.")
st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.8em; margin-bottom: 1em;">
        <img src="news-reccommender-project-logo.png" width="40" height="40" style="margin-bottom: 0;">
        <h1 style="margin: 0;">AI‑Powered News Recommender</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        h3 { color: #333366; margin-top: 2em; }
        h4 { font-size: 1.1em; margin-bottom: 0.3em; margin-top: 0.5em; line-height: 1.4; }
        .stButton>button { padding: 0.3em 0.6em; font-size: 0.9em; margin-right: 0.2em; }
        button {
            background-color: white;
            border: 1px solid #ddd;
            cursor: pointer;
        }
        button:hover {
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

API_KEY = "bfe5bfa038cb47e88389cd54c405d83c"
BASE_URL = "https://newsapi.org/v2/top-headlines"

# Load spaCy NLP model once
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Load .env if present
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("gemini-pro")

gemini_model = load_gemini_model()

# Smarter summarizer using Gemini with fallback
def gemini_summarize(text):
    try:
        prompt = f"Summarize this news article description in 1-2 sentences:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        doc = nlp(text)
        sentences = list(doc.sents)
        return sentences[0].text if sentences else "No summary available."

def analyze_sentiment(text):
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment
        return sentiment.score
    except Exception:
        return 0  # Default to neutral if analysis fails

def sentiment_to_emoji(score):
    if score > 0.25:
        return "😊"
    elif score < -0.25:
        return "😞"
    else:
        return "😐"

# Title and Instructions
st.write("Type in your interests (e.g., ‘AI, cricket, movies’), then hit **Get News** to receive top headlines.")

# User input
user_input = st.text_input("Your interests:")
sentiment_filter = st.checkbox("👍 Only show positive/neutral news")
run_sentiment = st.checkbox("🧠 Run sentiment analysis (uses NLP and may slow down)", value=True)

# Keyword-to-category mapping
interest_map = {
    # Sports
    "cricket": "sports", "football": "sports", "soccer": "sports", "tennis": "sports",
    # Entertainment
    "movie": "entertainment", "film": "entertainment", "bollywood": "entertainment",
    "music": "entertainment", "celebrity": "entertainment",
    # Technology
    "technology": "technology", "tech": "technology", "ai": "technology",
    "artificial": "technology", "intelligence": "technology", "machine": "technology",
    "gadgets": "technology",
    # Science
    "science": "science", "space": "science", "nasa": "science",
    # Health
    "health": "health", "covid": "health", "medicine": "health", "wellness": "health",
    # Business
    "business": "business", "economy": "business", "market": "business",
    # General/Politics
    "politics": "general", "global": "general", "election": "general", "government": "general"
}

# Extract matching categories using NLP
selected_categories = []
if user_input:
    doc = nlp(user_input)
    seen = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            lemma = token.lemma_.lower()
            cat = interest_map.get(lemma)
            if cat and cat not in seen:
                selected_categories.append(cat)
                seen.add(cat)

# Show matched categories summary
if selected_categories:
    st.markdown("### 🎯 Matched Categories:")
    st.markdown(" • ".join([f"[{cat.capitalize()}](#{cat})" for cat in selected_categories]), unsafe_allow_html=True)

# Main logic
if st.button("Get News"):
    if not selected_categories:
        st.warning("⚠️ No matching interests found. Try different keywords!")
    else:
        for category in selected_categories:
            st.markdown(f'<h3 id="{category}">🗞️ Top {category.capitalize()} News</h3>', unsafe_allow_html=True)
            params = {
                "apiKey": API_KEY,
                "category": category,
                "pageSize": 5
            }
            response = requests.get(BASE_URL, params=params).json()
            if response.get("status") == "ok" and response.get("articles"):
                for article in response["articles"]:
                    summary = gemini_summarize(article["description"])
                    sentiment_score = analyze_sentiment(article["description"]) if run_sentiment else 0
                    if sentiment_filter and sentiment_score < -0.25:
                        continue
                    emoji = sentiment_to_emoji(sentiment_score) if run_sentiment else ""
                    st.markdown(f"<h4>{emoji} {article['title']}</h4>", unsafe_allow_html=True)
                    st.write(f"📅 Published: {article.get('publishedAt', 'Unknown Date')}")
                    st.write(f"🔗 [Read full article]({article['url']})")

                    if article.get("description"):
                        st.markdown(f"📝 **Summary:** {summary}")
                    else:
                        st.markdown("📝 **Summary:** Not available.")

                    st.markdown(f'''
  <style>
      .like-button {{
          font-size: 1.1em;
          padding: 0.4em 0.6em;
          border-radius: 8px;
          background-color: white;
          border: 2px solid #ccc;
          margin-right: 0.3em;
          cursor: pointer;
          transition: background-color 0.2s, border-color 0.2s, transform 0.1s;
          display: inline-block;
          text-align: center;
      }}
      .like-button:hover {{
          background-color: #f0f0f0;
          border-color: #999;
          transform: scale(0.98);
      }}
      .like-button:active {{
          background-color: #e0e0e0;
          border-color: #666;
          transform: scale(0.95);
      }}
  </style>
  <div style="display: flex; gap: 0.5em;">
      <div class="like-button" onclick="this.classList.add('clicked')">👍</div>
      <div class="like-button" onclick="this.classList.add('clicked')">👎</div>
  </div>
  ''', unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.warning(f"No news found for category: {category}")

# Footer
st.markdown("---")
st.markdown("""
<center>
Built by Saksham Virmani | Powered by [NewsAPI](https://newsapi.org) & spaCy | Deployed on GCP
</center>
""", unsafe_allow_html=True)