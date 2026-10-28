import streamlit as st
import joblib
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.joblib")

model = load_model()
st.title("📰 Fake News Detection")
st.write(
    "Enter a news headline or article to get "
    "an ML-based classification."
)

article = st.text_area(
    "News article",
    height=220,
    placeholder="Paste the news article here..."
)

if st.button("Analyze News", type="primary"):
    if not article.strip():
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([article])[0]

        if prediction == 0:
            st.error("Prediction: Likely Fake")
        else:
            st.success("Prediction: Likely Real")

        