import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load and preprocess data
movie_data = pd.read_csv('your_data.csv', encoding='ISO-8859-1')
movie_data = movie_data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]
movie_data = movie_data.dropna(subset=['Rating'])

# Feature and target separation
X = movie_data.drop('Rating', axis=1)
y = movie_data['Rating']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing and pipeline
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model (can be skipped if using a pre-trained model)
model.fit(X_train, y_train)

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
        # Create a new dataframe for the user input
        user_input = pd.DataFrame({
            'Genre': [genre],
            'Director': [director],
            'Actor 1': [actor_1],
            'Actor 2': [actor_2],
            'Actor 3': [actor_3]
        })
        
        # Make prediction
        predicted_rating = model.predict(user_input)
        st.success(f'Predicted Rating: {predicted_rating[0]:.2f}')
    else:
        st.error('Please fill all fields to predict the rating.')

# Model evaluation section
st.write('---')
st.subheader('Model Evaluation')

# Display the RMSE of the trained model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
st.write(f'Root Mean Squared Error (RMSE) on test data: {rmse:.2f}')
