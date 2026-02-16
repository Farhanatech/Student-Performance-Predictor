# Student Performance Predictor

## Project Description
This is a full-stack learning project designed to predict a student's final exam score based on weekly study hours, attendance percentage, and past academic performance. The project demonstrates the integration of a Machine Learning model with a web-based user interface.

## Technical Features
* Data Preprocessing: Implementation of the Interquartile Range (IQR) method to identify and remove outliers from the training data.
* Machine Learning Model: Utilizes a Linear Regression algorithm enhanced with Polynomial Features (Degree 3) to capture non-linear relationships.
* Backend: A Flask-based REST API that handles incoming data and returns model predictions in real-time.
* Frontend: A responsive web interface built with HTML5, CSS3, and Vanilla JavaScript using the Fetch API.

## Project Structure
* backend/: Contains the Flask application (app.py), the serialized model (student_performance_model.pkl), and the dependencies list (requirements.txt).
* frontend/: Contains the user interface files including index.html for the structure and script.js for the API logic.
* .gitignore: Configured to prevent large or temporary files like virtual environments (venv) and cache from being uploaded.

## Installation and Usage

### Backend Setup
1. Navigate to the backend directory:
   cd backend
2. Create a virtual environment:
   python -m venv venv
3. Activate the environment:
   - Windows: venv\Scripts\activate
   - Mac/Linux: source venv/bin/activate
4. Install required libraries:
   pip install -r requirements.txt
5. Start the server:
   python app.py

### Frontend Setup
1. Ensure the backend server is running on [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Open the frontend/index.html file in any modern web browser.
3. Enter the student data and click the predict button to see the results.

## Learning Objectives
The primary goal of this project was to learn the end-to-end workflow of a Machine Learning project, specifically focusing on how to serve a trained model through a web server and handle Cross-Origin Resource Sharing (CORS) between a frontend and a backend.
