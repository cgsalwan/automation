"""
Rental Price Prediction — Decoupled Model Deployment

Orchestration notebook: loads the trained model, creates the Hugging Face
Spaces, uploads the backend_files/ and frontend_files/ directories, and
calls the deployed API for online and batch inference.

The actual deployed application code lives in:
  backend_files/app.py, requirements.txt, Dockerfile   (Flask API)
  frontend_files/app.py, requirements.txt, Dockerfile  (Streamlit UI)
"""

# # Rental Price Prediction — Decoupled Model Deployment
# 
# *Deploying a trained pricing model as a containerized, decoupled web application (Flask backend + Streamlit frontend).*

# <center><img src="https://www.vectorlogo.zone/logos/airbnb/airbnb-ar21.svg" width="720"></center>
# 
# <center><b><font size='5'>Airbnb Rental Price Prediction</font></b></center>

# # Problem Statement

# ## Business Context

# Short-term rental platforms need to help hosts set competitive, data-driven prices for their listings. A pricing model that only runs inside a notebook isn't useful in practice — it needs to be deployed as a service that hosts and internal tools can query in real time, and that can scale independently as demand grows.
# 
# This project takes a previously trained rental price prediction model and deploys it as a decoupled, containerized web application: a Flask API serving predictions, and a separate Streamlit frontend for interactive use.

# ## Objective

# The goal is to deploy the rental price prediction model as a containerized, decoupled web application for reliable, real-time use. The solution should:
# 
# - **Enable real-time predictions:** let users input property details and get a price prediction instantly.
# - **Be scalable:** use a containerized, microservice-based architecture so components can scale independently.
# - **Decouple frontend and backend:** serve predictions via a Flask API, with a separate Streamlit frontend consuming that API, so either side can be updated or scaled without touching the other.
# - **Be portable across environments:** use Docker to package the model and its dependencies for consistent deployment, in this case to Hugging Face Spaces.

# # Deployment Approach

# The crux of our solution is to decouple the frontend and backend of the application for better accessibility and seamless integration with other services. This modular design enhances maintainability and scalability.
# 
# **Backend Development (Flask):**
# 
# 1. We will develop a Flask application (app.py) responsible for:
#   - Loading the serialized XGBoost model we trained and saved as rental_price_prediction_model_v1_0.joblib.
#   - Exposing two API endpoints:
#     - `/v1/rental`: For predicting the rental price of a single property, accepting input features as JSON, mirroring the data format used during training and EDA.
#     - `/v1/rentalbatch`: For batch predictions on multiple properties, processing data uploaded as a CSV file, consistent with the dataset structure.
# 2. This backend handles the core prediction logic, applying the same data preprocessing steps (imputation, scaling, and encoding) used during model training to ensure consistency and accuracy. It returns predictions as JSON responses, facilitating easy integration with the frontend.
# 3. We'll deploy this Flask app, the serialized model, and the requirements.txt file to a Hugging Face Space using a Dockerfile. This makes the prediction service publicly accessible via a unique URL.
# 
# 
# **Frontend Development (Streamlit):**
# 
# 1. A separate Streamlit application (app.py) will serve as the user interface, tailored for Airbnb hosts and internal teams.
# 
# 2. This frontend will include:
#   - Form-based inputs for online predictions, allowing users to enter property details like room type, accommodates, bathrooms, etc., aligning with the features used in the model.
#   - A CSV file uploader for batch predictions, enabling users to analyze multiple properties simultaneously, similar to the dataset format used for training.
# 3. The Streamlit app will use the `requests` library to communicate with the Flask API, sending input data and displaying predictions in a user-friendly format. This interaction facilitates a seamless user experience.
# 4. Similar to the backend, we'll deploy the Streamlit app to a Hugging Face Space using a `Dockerfile` with its own `requirements.txt` for managing dependencies.
# 
# Once deployed, Airbnb stakeholders can access the application through the frontend URL. They can input property details or upload a CSV file to obtain instant rental price predictions, enabling data-driven pricing strategies. This decoupled approach ensures flexibility and ease of access for Airbnb users.

# # Load the Serialized Model

# **Note:** To ensure continuity and leverage our previous work, we'll utilize the serialized XGBoost model trained and saved in the previous week.

import os

# Create a folder to upload your trained serialized model into it
os.makedirs("backend_files", exist_ok=True)

# - We need to now upload the serialized model (`rental_price_prediction_model_v1_0.joblib`) into the `backend_files` folder.
# - Once uploaded, we will load this model into our application for generating rental price predictions.
# 
# - This approach allows us to seamlessly integrate the pre-trained model into our deployment workflow, eliminating the need for retraining.

# Define the file path to load the uploaded serialized model
model_path = "rental_price_prediction_model_v1_0.joblib"

import joblib

# Load the saved model pipeline from the file
saved_model = joblib.load(model_path)

# Confirm the model is loaded
print("Model loaded successfully.")

saved_model

# Let's try making predictions on the batch dataset `(airbnb_rental_batch_data.csv)` using the deserialized model.
# - Please ensure that the saved model `(rental_price_prediction_model_v1_0.joblib)` is loaded before making predictions.
# - We will apply `np.exp` to the predictions to convert the log prices into actual prices.

import pandas as pd
import numpy as np

# Load the data:
airbnb_batch_data = pd.read_csv("airbnb_rental_batch_data.csv")

# Make predictions (fet log_prices)
predictions_log_prices = saved_model.predict(airbnb_batch_data)

# Convert log prices to actual prices
predictions_actual_prices = np.exp(predictions_log_prices)

# Display predictions:
print(predictions_actual_prices)

# - As we can see, the model can be directly used for making predictions without any retraining.

# # App Backend

# ## Setting up a Hugging Face Docker Space for the Backend

# - We are creating a Hugging Face Docker Space for our backend using the Hugging Face Hub API.
# - This automates the space creation process and enables seamless deployment of our Flask app.

# Import the login function from the huggingface_hub library
from huggingface_hub import login

# Login to your Hugging Face account using your access token
# Replace "YOUR_HUGGINGFACE_TOKEN" with your actual token
login(token="YOUR_HUGGINGFACE_TOKEN")

# Import the create_repo function from the huggingface_hub library
from huggingface_hub import create_repo

# Try to create the repository for the Hugging Face Space
try:
    create_repo("RentalPricePredictionBackend",  # One can replace "Backend_Docker_space" with the desired space name
        repo_type="space",  # Specify the repository type as "space"
        space_sdk="docker",  # Specify the space SDK as "docker" to create a Docker space
        private=False  # Set to True if you want the space to be private
    )
except Exception as e:
    # Handle potential errors during repository creation
    if "RepositoryAlreadyExistsError" in str(e):
        print("Repository already exists. Skipping creation.")
    else:
        print(f"Error creating repository: {e}")

# ## Flask Web Framework

# (contents of backend_files/app.py — see that file directly)

# ## Dependencies File

# (contents of backend_files/requirements.txt — see that file directly)

# ## Dockerfile

# (contents of backend_files/Dockerfile — see that file directly)

# ## Uploading Files to Hugging Face Space for the Backend

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

repo_id = "your-username/RentalPricePredictionBackend"  # Your Hugging Face space id

# Initialize the API
api = HfApi()

# Upload Streamlit app files stored in the folder called deployment_files
api.upload_folder(
    folder_path="/content/backend_files",  # Local folder path
    repo_id=repo_id,  # Hugging face space id
    repo_type="space",  # Hugging face repo type "space"
)

# # App Frontend

# ## Setting up a Hugging Face Docker Streamlit Space for the Frontend

# ## Points to note before executing the below cells
# - Create a Streamlit space on Hugging Face by following the instructions provided on the content page titled **`Creating Spaces and Adding Secrets in Hugging Face`** from Week 1

# ## Streamlit for Interactive UI

# Create a folder for storing the files needed for frontend UI deployment
os.makedirs("frontend_files", exist_ok=True)

# (contents of frontend_files/app.py — see that file directly)

# ## Dependencies File

# (contents of frontend_files/requirements.txt — see that file directly)

# ## Dockerfile

# (contents of frontend_files/Dockerfile — see that file directly)

# ## Uploading Files to Hugging Face Space for the Frontend

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

repo_id = "--------------------------"  # Your Hugging Face space id

# Initialize the API
api = HfApi()

# Upload Streamlit app files stored in the folder called deployment_files
api.upload_folder(
    folder_path="/content/frontend_files",  # Local folder path
    repo_id=repo_id,  # Hugging face space id
    repo_type="space",  # Hugging face repo type "space"
)

# # Inferencing using Flask API

# As the ***frontend and backend are decoupled***, we can ***access the backend directly for predictions***.
# - The decoupling ensures seamless interaction with the deployed model while leveraging the API for scalable inference.

# Let's see how to interact with the Flask API programatically within this notebook to perform **online** and **batch inference**.
# 
# We will
# 1. Send API requests for both online and batch inference.
# 2. Process and check the model predictions.

import json  # To handle JSON formatting for API requests and responses
import requests  # To send HTTP requests to the deployed Flask API

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations

model_root_url = "https://<username>-<repo_id>.hf.space"  # Base URL of the deployed Flask API on Hugging Face Space; enter user name and space name before running the cell

model_url = model_root_url + "/v1/rental"  # Endpoint for online (single) inference

# Since our model predictions are served through the Flask endpoint we created, we need to call this endpoint to make a prediction.
# 
# > ```@app.post('/v1/rental')```

model_batch_url = model_root_url + "/v1/rentalbatch"  # Endpoint for batch inference

# > ```@app.post('/v1/rentalbatch')```

# ## Online Inference

# The idea is to send a single request to the API and receive an immediate response. This is useful for real-time applications like recommendation systems and fraud detection.
# 
# * This data is sent as a JSON payload in a POST request to the model endpoint.
# * The model processes the input features and returns a prediction as a JSON payload.

payload = {
  "room_type": "Entire home/apt",
  "accommodates": 5,
  "bathrooms": 3,
  "cancellation_policy": "strict",
  "cleaning_fee": True,
  "instant_bookable": "f",
  "review_scores_rating": 90,
  "bedrooms": 3,
  "beds": 3
}

# This payload dictionary includes all the necessary features in the expected
# format for online (single property) prediction, ensuring consistency with the
# model's training data.

# Sending a POST request to the model endpoint with the test payload
response = requests.post(model_url, json=payload)

response

# - The `<Response [200]>` you see is an HTTP status code.
# - It indicates that your request was successful, and the server was able to process it without any problems.

print(response.json())

# ## Batch Inference

# The idea is to send a batch of requests to the API and receive a response. The backend reads the entire dataset, runs it through the ML model, and returns the prediction for every row in the file. This is useful for applications like loan default prediction and customer churn prediction, where we don't need results instantaneously.
# 
# * This data is sent as a CSV file in a POST request to the model endpoint.
# * The model processes each row containing the input features and returns the predictions for each row as one single JSON payload.

import pandas as pd

# Load the sample batch data for Airbnb
airbnb_batch_data = pd.read_csv("airbnb_rental_batch_data.csv")

# - The model was trained using certain set of numerical and categorical features before being serialized.
# - We need to use the same set of features and pass the data to the API in order to get predictions.
# - We define these feature lists below, where we select the necessary columns from the batch data to ensure the model receives the expected input format for prediction.

# List of numerical features in the Airbnb dataset
numeric_features = [
    'id',
    'accommodates',
    'bathrooms',
    'review_scores_rating',
    'bedrooms',
    'beds'
]

# List of categorical features in the Airbnb dataset
categorical_features = [
    'room_type',
    'cancellation_policy',
    'cleaning_fee',
    'instant_bookable'
]

# Define predictor matrix (X) using selected numeric and categorical features
batch_input_data = airbnb_batch_data[numeric_features + categorical_features]

# Prepare batch input for API request
batch_input = {
    'file': batch_input_data.to_csv(header=True, index=False).encode('utf-8')
}

# Send request to the model API for batch predictions
response = requests.post(
    model_batch_url,  # Model endpoint URL
    files=batch_input
)

response

response.text

# - As we can see, we receive a JSON where each key represents a property ID, and the value represents the model's predicted rental price (in dollars) for that property.

# # Conclusion

# 1. **Flexibility and Scalability**: By separating the frontend and backend, we can easily update or scale each component independently. This means we can make changes to the user interface without affecting the prediction model, or vice versa. This also allows us to handle a large number of requests by scaling the backend without impacting the frontend's performance. It's like having a system with changeable parts, making it more adaptable and robust.
# 
# 2. **Technology Agnostic**: The decoupled architecture allows us to use different technologies for the frontend and backend. For example, we can use Streamlit for the frontend and Flask for the backend, or any other suitable technologies. This flexibility enables us to choose the best tools for the job at hand.
# 
# 3. **Reusability**: The backend API can be reused by other applications or services. This means we can integrate the prediction functionality into different parts of Airbnb's platform or even share it with external partners. This fosters greater efficiency and integration possibilities, extending the model's benefits beyond a single application. It's like creating a versatile tool that can be used in various projects, maximizing its value.

# <font size=6 color="blue">Power Ahead!</font>
# ___
