from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()

def preprocess_text(review):
    review = review.lower().split()
    encoded_review = [word_index.get(word , 2) + 3 for word in review]
    padded_review = pad_sequences([encoded_review] , maxlen = 500)

    return padded_review


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')
user_input = st.text_area('Movie Review')
model = load_model('imdb_rnn.keras')

if st.button('Classify'):
    processed_input = preprocess_text(user_input)
    predict_proba = model.predict(processed_input)[0][0]
    sentiment = 'Postive' if predict_proba > .5 else 'Negative'

    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score : {predict_proba}')
else:
    st.write('Write a movie review')
