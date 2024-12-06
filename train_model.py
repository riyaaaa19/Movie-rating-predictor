import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset (adjust the file path and encoding as necessary)
data = pd.read_csv('C:\\Users\\riyas\\OneDrive\\Desktop\\movie_predictor\\your_data.csv', encoding='ISO-8859-1')

# Clean the column names by stripping any extra spaces
data.columns = data.columns.str.strip()

# Check if 'Language' is a categorical column and encode it using LabelEncoder
if 'Language' in data.columns:
    label_encoder = LabelEncoder()
    data['Language'] = label_encoder.fit_transform(data['Language'].astype(str))

# Handle missing values by imputing them
imputer = SimpleImputer(strategy='most_frequent')  # Use the most frequent value for imputation
data_imputed = data.copy()
data_imputed[data_imputed.columns] = imputer.fit_transform(data_imputed)

# Split the dataset into features (X) and target (y)
target = 'Rating'  # Change this if your target column has a different name
X = data_imputed.drop(columns=[target])
y = data_imputed[target]

# Identify columns with categorical data (non-numeric columns)
categorical_columns = X.select_dtypes(include=['object']).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['float64', 'int64']).columns),  # Impute missing numeric values
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # OneHotEncode categorical columns
    ]
)

# Create a pipeline that includes both preprocessing and the RandomForestRegressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the pipeline
model_pipeline.fit(X_train, y_train)

# Predict the target values using the trained model on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate mean squared error for regression
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Optional: Display feature importance (after fitting)
regressor = model_pipeline.named_steps['regressor']
feature_importance = regressor.feature_importances_

# Print feature importance if needed
print("Feature Importance:", feature_importance)

# Optional: Visualize feature importance using a bar plot
import matplotlib.pyplot as plt

# Use the encoded feature names for visualization
encoded_columns = model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_columns)
all_columns = list(X.select_dtypes(include=['float64', 'int64']).columns) + list(encoded_columns)

plt.barh(all_columns, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
