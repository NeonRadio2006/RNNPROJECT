import streamlit as st
from predict import predict_sentiment
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🍿")
st.title("🍿 IMDB Movie Review Sentiment Analyzer")
st.write("This application uses a Simple Recurrent Neural Network (RNN) trained on the IMDB dataset to classify movie reviews as **Positive** or **Negative**.")
user_input=st.text_area(
    "Enter your movie review here:", 
    height=150, 
    placeholder="e.g., The cinematography was stunning, but the plot was incredibly boring..."
)
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing review..."):
            sentiment,confidence=predict_sentiment(user_input)
            if "Error" in sentiment:
                st.error("Model file not found. Have you run `python train.py` yet?")
            else:
                st.markdown("---")
                st.subheader("Analysis Result")
                if sentiment == "Positive":
                    st.success(f"**Sentiment:** {sentiment} 🤩")
                else:
                    st.error(f"**Sentiment:** {sentiment} 😞")
                st.info(f"**Model Confidence:** {confidence:.2%}")
                st.progress(float(confidence))