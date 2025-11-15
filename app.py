import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords # Import here for type hints

# --- NLTK Setup: Use st.cache_resource to download data once ---
@st.cache_resource
def download_nltk_data():
    """Ensures NLTK stopwords are downloaded once in the cloud environment."""
    try:
        nltk.download('stopwords', quiet=True)
        return True, set(stopwords.words('english'))
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        return False, set()

# Perform the download and get stopwords
NLTK_LOADED, STOP_WORDS = download_nltk_data()

# --- Constants ---
MODEL_FILE = 'pac_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

# --- Utility Functions ---

def clean_text(text):
    """Cleans the input text using the same steps applied during model training."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Uses the globally defined STOP_WORDS set
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

@st.cache_resource
def load_artifacts():
    """Loads the trained model and vectorizer only once."""
    try:
        pac = joblib.load(MODEL_FILE)
        tfidf_vectorizer = joblib.load(VECTORIZER_FILE)
        return pac, tfidf_vectorizer, None
    except FileNotFoundError:
        error_message = (
            f"**Error: Model files not found!** "
            f"Ensure `{MODEL_FILE}` and `{VECTORIZER_FILE}` are uploaded to GitHub."
        )
        return None, None, error_message
    except Exception as e:
        return None, None, f"An unexpected error occurred while loading artifacts: {e}"

# Load the artifacts
pac, tfidf_vectorizer, load_error = load_artifacts()

# --- Main Streamlit Application ---

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )

    st.title("📰 Real or Fake? News Detector")
    st.markdown("---")

    # Display warning if NLTK failed to load
    if not NLTK_LOADED:
        st.warning(
            "⚠️ **Warning:** NLTK stopwords failed to load. "
            "The text cleaning step is suboptimal."
        )
        
    # --- Error Handling for Model Loading ---
    if load_error:
        st.error(load_error)
        st.stop() 

    # --- Prediction Function ---
    def predict_fake_news(news_text, model, vectorizer):
        # 1. Clean the text
        cleaned_text = clean_text(news_text)

        # 2. Vectorize the text
        vectorized_text = vectorizer.transform([cleaned_text])

        # 3. Predict the label (0: Fake, 1: Real)
        prediction = model.predict(vectorized_text)[0]

        return prediction

    # --- UI Components ---
    st.header("Enter the News Article Text Below")
    st.info("Paste the full title and body text of the article for the best results.")

    # Text input area
    news_input = st.text_area(
        "News Content",
        placeholder="Paste your news article here...",
        height=300
    )

    # Classification button
    if st.button("Classify Article", use_container_width=True, type="primary"):
        if news_input:
            with st.spinner('Analyzing content...'):
                prediction = predict_fake_news(news_input, pac, tfidf_vectorizer)

            st.markdown("### Classification Result")

            if prediction == 1:
                st.balloons()
                st.success(
                    "**REAL NEWS (Label 1) ✅**"
                )
                st.markdown("The model has high confidence that this article is likely legitimate.")
            else:
                st.error(
                    "**FAKE NEWS (Label 0) ❌**"
                )
                st.markdown("The model suggests this content has characteristics commonly found in fabricated or unreliable news.")
        else:
            st.warning("Please paste some text into the box above to analyze.")

    st.markdown("---")
    st.caption("Powered by TF-IDF Vectorization and Passive Aggressive Classifier.")

if __name__ == "__main__":
    main()