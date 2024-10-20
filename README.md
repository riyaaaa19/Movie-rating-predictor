# Movie Rating Predictor

This project is a **Movie Rating Predictor** built using **Streamlit** and **Random Forest Regressor** from **Scikit-learn**. The application predicts the rating of a movie based on various features like genre, director, and cast.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features
- Predict movie ratings based on user input.
- User-friendly interface to enter movie details.
- Utilizes machine learning to provide predictions.

## Technologies Used
- **Python**
- **Streamlit**: For building the web application.
- **Scikit-learn**: For machine learning model and preprocessing.
- **Pandas**: For data manipulation and analysis.
- **Joblib**: For saving and loading the model.
- **Git**: For version control.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/riyaaaa19/Movie-rating-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Movie-rating-predictor
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Run the Streamlit application:
```bash
streamlit run streamlit_movie_rating.py
```
2. Open your browser and go to http://localhost:8501 to interact with the application.
3. Enter the movie details in the input fields and click the Predict Rating button to see the predicted rating.
## How It Works
1.The model uses a Random Forest Regressor to predict movie ratings based on input features like genre, director, and actors.
2. Data preprocessing is performed using Pandas and Scikit-learn to handle missing values and categorical variables.
3. The trained model is saved using Joblib for future use.
## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please create a pull request or open an issue.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

### Steps to Add the README

1. **Create a New File**: In your GitHub repository, click on **"Add file"** > **"Create new file."**
2. **Name the File**: Name it `README.md`.
3. **Copy and Paste**: Copy the contents of the sample README above and paste it into the new file.
4. **Commit the Changes**: Scroll down, add a commit message, and click **"Commit new file."**

### Customize as Needed

Feel free to adjust any sections to better reflect your project specifics or preferences. If you need further assistance or want to add more details, just let me know!
