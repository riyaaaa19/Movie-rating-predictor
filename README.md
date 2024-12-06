
---

# Movie Predictor

**Movie Predictor** is a machine learning project that uses the Random Forest Regressor to predict movie ratings based on various movie features. This project is designed for beginners to help understand the steps involved in training a model and making predictions.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

To set up and run this project, you'll need Python and some dependencies. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/riyaaaa19/movie_predictor.git
    cd movie_predictor
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset

The dataset used in this project contains movie data, including features such as the movie name, genre, language, and ratings. The target variable is the `Rating`, which is predicted by the model based on the movie features.

Make sure your dataset contains columns like:
- `Movie Name`
- `Genre`
- `Language`
- `Rating` (this is the target column for prediction)

---

## Usage

This project is divided into two parts: **training the model** and **making predictions**.

### Model Training

1. **Prepare the Dataset**: Place your dataset file (e.g., `movie_data.csv`) in the project folder.
2. **Train the Model**: Run the `train_model.py` script to train the Random Forest Regressor model.
    ```bash
    python train_model.py
    ```
    This script will:
    - Load and preprocess the data.
    - Train the model using the movie features to predict ratings.
    - Save the trained model for later use in the prediction script.

### Making Predictions

After training the model, you can use the `streamlit_movie_rating.py` script to predict movie ratings on new or unseen data.

    ```
    The script will:
    - Load the trained model saved by the `train_model.py` script.
    - Load the new data and preprocess it.
    - Predict ratings for each movie and print the results.

---

## Model Training

In this project, we use a **Random Forest Regressor** model to predict movie ratings.

- **Data Preprocessing**: The data is cleaned by handling missing values, encoding categorical columns, and splitting it into training and test sets.
- **Training**: The model is trained using the training data (80% of the total data) and then evaluated on the test data (20% of the total data).
- **Saving the Model**: Once trained, the model is saved using `joblib` to make predictions later.

---

## Making Predictions

The `streamlit_movie_rating.py` script uses the trained model to predict ratings for new movies. Here's how it works:
- The new data is preprocessed in the same way as the training data.
- The trained model is loaded, and predictions are made based on the input features.
- Predicted ratings are displayed for each movie in the new dataset.

---

## Evaluation

The model's performance is evaluated using **Mean Squared Error (MSE)**. A lower MSE indicates better performance.

Example output:
```bash
Mean Squared Error: 0.42
```

This score represents the difference between the actual and predicted ratings, with the goal of minimizing this value.

---

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests.

### Steps to contribute:
1. Fork the repository
2. Clone your forked repository
3. Create a new branch
4. Make your changes and commit them
5. Push the changes to your forked repository
6. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

.
