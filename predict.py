import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
MAX_FEATURES=10000
MAX_LEN=500
try:
    model=tf.keras.models.load_model('simple_rnn_imdb.h5')
except Exception as e:
    model=None
    print("Warning:simple_rnn_imdb.h5 not found.Please run train.py first.")
word_index=tf.keras.datasets.imdb.get_word_index()
def preprocess_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    words=text.split()
    sequence=[]
    for word in words:
        idx=word_index.get(word)
        if idx is not None and (idx+3)<MAX_FEATURES:
            sequence.append(idx+3)
        else:
            sequence.append(2) 
    padded_sequence=pad_sequences([sequence],maxlen=MAX_LEN)
    return padded_sequence
def predict_sentiment(text):
    if model is None:
        return "Error: Model not found.", 0.0
    processed_text=preprocess_text(text)
    prediction_prob=model.predict(processed_text)[0][0]
    sentiment="Positive" if prediction_prob >= 0.5 else "Negative"
    confidence=prediction_prob if sentiment == "Positive" else (1-prediction_prob)
    return sentiment, confidence