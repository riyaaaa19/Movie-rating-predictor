import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('movie_rating_model.pkl')  # Load the trained model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained first.")
        return None

# Streamlit app layout
st.title('Movie Rating Predictor')

# Input form for user to provide details
st.write("Enter the details of the movie to predict its rating:")

genre = st.text_input('Genre')
director = st.text_input('Director')
actor_1 = st.text_input('Actor 1')
actor_2 = st.text_input('Actor 2')
actor_3 = st.text_input('Actor 3')

# Predict button
if st.button('Predict Rating'):
    if genre and director and actor_1 and actor_2 and actor_3:
        # Load the pre-trained model
        model = load_model()

        if model:
            # Create a new dataframe for the user input
            user_input = pd.DataFrame({
                'Genre': [genre],
                'Director': [director],
                'Actor 1': [actor_1],
                'Actor 2': [actor_2],
                'Actor 3': [actor_3]
            })

            # Preprocess and make prediction
            predicted_rating = model.predict(user_input)
            st.success(f'Predicted Rating: {predicted_rating[0]:.2f}')
    else:
        st.error('Please fill all fields to predict the rating.')

