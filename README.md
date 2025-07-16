✈️ Flight Delay Prediction Web App

A web-based application built using Flask, scikit-learn, and Matplotlib, designed to predict whether a flight will be delayed based on several operational and environmental 
features. It also provides a delay trend visualization by flight origin.

🚀 Features

🔍 Flight Delay Prediction using a trained RandomForestClassifier
📊 Dynamic line chart visualizing delays by origin airport
🧠 Incorporates SMOTE to handle class imbalance
🎯 Encodes categorical data using LabelEncoder
💾 Stores trained model and encoders as pickle files for reuse
🖼 Simple and responsive frontend with HTML + CSS

🧠 Machine Learning Pipeline

Model: Random Forest Classifier
Imbalance Handling: SMOTE
Encoders: Label Encoding for categorical features
Training Data: Loaded from flight_datafinal.csv

🛠 How It Works

User enters flight details via the web form.
Input data is label-encoded using pre-trained encoders.
The trained model (RandomForestClassifier) predicts if the flight will be Delayed or On Time.
A line graph showing total delays by origin airport is generated and displayed alongside the result.



