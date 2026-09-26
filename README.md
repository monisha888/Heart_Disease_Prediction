❤️ Heart Disease Prediction using Machine Learning



📌 Project Overview

Heart Disease Prediction is a Machine Learning project that predicts whether a person is likely to have heart disease based on selected medical parameters.

The project includes data preprocessing, feature selection, data balancing, feature scaling, multiple machine learning algorithms, model evaluation, model saving, and a Flask web application.

This project was developed as part of the Vihara Tech learning/internship project.

🚀 Live Deployment

🔗 Live Demo: Click here to use the Heart Disease Prediction App

Replace YOUR_DEPLOYMENT_LINK with your deployed application URL.

🎯 Objective

The objective of this project is to build a Machine Learning classification system that predicts whether heart disease is detected based on patient input features.

🧠 Machine Learning Workflow

Dataset
   ↓
Data Loading
   ↓
Train-Test Split
   ↓
Yeo-Johnson Transformation
   ↓
Outlier Handling
   ↓
Feature Selection
   ↓
SMOTE Data Balancing
   ↓
Standard Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Gaussian Naive Bayes Model
   ↓
Save Model + Scaler
   ↓
Flask Web Application
   ↓
Heart Disease Prediction

📂 Project Structure

Heart-Disease-Prediction/
│
├── heart.csv
├── main.py
├── variable_transformation.py
├── feature_selection.py
├── all_models.py
├── log.py
├── app.py
├── model.pkl
├── scaled.pkl
├── requirements.txt
├── Procfile
├── logo.png
├── templates/
│   └── index.html
├── logs/
│   └── main.log
└── README.md

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

SciPy

Imbalanced-learn

XGBoost

Matplotlib

Seaborn

Flask

🔄 Data Preprocessing

1. Train-Test Split

The dataset is divided into training and testing data using an 80:20 split.

2. Yeo-Johnson Transformation

Yeo-Johnson transformation is applied to the input features before outlier handling.

3. Outlier Handling

The project uses the Interquartile Range (IQR) method:

IQR = Q3 - Q1
Upper Limit = Q3 + 1.5 × IQR
Lower Limit = Q1 - 1.5 × IQR

Values outside the limits are trimmed to the corresponding boundary values.

4. Feature Selection

Variance Threshold is used for constant and quasi-constant feature selection.

5. SMOTE

SMOTE is used to balance the training dataset.

6. Standard Scaling

StandardScaler is used to scale the features before model training.

🤖 Machine Learning Models

The project compares the following classification algorithms:

Model

Algorithm

KNN

K-Nearest Neighbors

NB

Gaussian Naive Bayes

LR

Logistic Regression

DT

Decision Tree

RF

Random Forest

ADA

AdaBoost

GB

Gradient Boosting

XGB

XGBoost

The models are evaluated using:

Accuracy

Confusion Matrix

Classification Report

🏆 Final Model

Gaussian Naive Bayes is used as the final prediction model.

The trained model is saved as:

model.pkl

The fitted scaler is saved as:

scaled.pkl

These files are loaded by the Flask application for making predictions.

🌐 Flask Web Application

The Flask application provides a user-friendly interface for entering patient information and receiving a prediction.

Input Features

age
sex
cp
thalach
oldpeak
slope
thal

Prediction Results

The application displays:

❤️ Heart Disease Detected

or

💚 No Heart Disease

🚀 Installation

1. Clone the Repository

git clone https://github.com/yourusername/Heart-Disease-Prediction.git

2. Open the Project

cd Heart-Disease-Prediction

3. Create Virtual Environment

python -m venv venv

4. Activate Virtual Environment

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

5. Install Requirements

pip install -r requirements.txt

▶️ Run the Machine Learning Pipeline

python main.py

This performs:

Data Loading
→ Preprocessing
→ Feature Selection
→ SMOTE
→ Scaling
→ Model Comparison
→ Model Training
→ Save model.pkl
→ Save scaled.pkl

🌐 Run the Flask Application

python app.py

Open the application in your browser:

http://127.0.0.1:5000/

📊 Model Evaluation

The project uses:

Accuracy

Measures the percentage of correctly classified observations.

Confusion Matrix

Shows:

True Positive

True Negative

False Positive

False Negative

Classification Report

Provides:

Precision

Recall

F1-score

Support

📝 Logging

The project includes a logging module to record information and errors during execution.

Log files can be stored inside the logs directory.

✨ Key Features

❤️ Heart disease prediction

🧹 Data preprocessing

📊 Feature selection

🔄 Outlier handling

⚖️ SMOTE data balancing

📏 Standard scaling

🤖 Multiple ML algorithms

📈 Model evaluation

💾 Model serialization

🌐 Flask web application

📝 Logging and exception handling

⚠️ Disclaimer

This project is developed for educational and demonstration purposes only. It should not be used as a substitute for professional medical diagnosis or medical advice.

👩‍💻 Author

Monisha G E

B.Tech – Computer Science and Engineering

🏢 Organization

Vihara Tech

Learn • Intern • Get Placed
