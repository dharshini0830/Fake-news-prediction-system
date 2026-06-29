import streamlit as st
import pickle
import re

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------
# LOAD MODEL & VECTORIZER
# ---------------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------------
# CLEAN TEXT FUNCTION
# ---------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.write("### Fake News Detection System")

st.sidebar.write("Algorithm: Logistic Regression")

st.sidebar.write("Feature Extraction: TF-IDF")

st.sidebar.write("Accuracy: 98.7%")

# ---------------------------------
# TITLE
# ---------------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Detect whether a news article is REAL or FAKE using Machine Learning and Natural Language Processing."
)

# ---------------------------------
# INPUT AREA
# ---------------------------------

news = st.text_area(
    "Paste News Article Here",
    height=250
)

# ---------------------------------
# PREDICT BUTTON
# ---------------------------------

if st.button("Check News"):

    if news.strip() == "":

        st.warning("Please enter a news article.")

    else:

        cleaned_news = clean_text(news)

        news_vector = vectorizer.transform(
            [cleaned_news]
        )

        prediction = model.predict(
            news_vector
        )

        probability = model.predict_proba(
            news_vector
        )

        confidence = max(
            probability[0]
        ) * 100

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.success(
                "✅ REAL NEWS"
            )

        else:

            st.error(
                "❌ FAKE NEWS"
            )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        st.write(
            "Word Count:",
            len(news.split())
        )

        if len(news.split()) < 30:

            st.warning(
                "Short news articles may produce less reliable predictions."
            )

# ---------------------------------
# HOW IT WORKS
# ---------------------------------

st.markdown("---")

st.subheader("📖 How It Works")

st.write("""
1. User enters a news article.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical features.
4. Logistic Regression predicts Real or Fake.
5. Confidence score is displayed.
""")

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.caption(
    "Developed using Python, NLP, TF-IDF, Logistic Regression and Streamlit."
)