from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend issues
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Handle class imbalance

app = Flask(__name__)

# File paths
DATA_FILE = "flight_datafinal.csv"
MODEL_FILE = "delay_model.pkl"
ENCODER_FILE = "label_encoders.pkl"

# Load dataset
if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)
else:
    data = pd.DataFrame(columns=['flight_number', 'scheduled_arrival', 'scheduled_departure', 'actual_departure',
                                 'month', 'origin', 'destination', 'weather', 'air_traffic_control',
                                 'airline_operations', 'airport_operations', 'passenger_related',
                                 'external_factors', 'delayed'])  # Empty DataFrame if file is missing

# Preprocessing function
def preprocess_data(df, training=True, label_encoders=None):
    """Preprocess dataset: Encode categorical values & handle missing values"""
    df = df.fillna("unknown")  # Fill missing values
    if label_encoders is None:
        label_encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':  # Ensure column is categorical
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                if training:
                    df[col] = label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = df[col].apply(lambda x: label_encoders[col].transform([x])[0] 
                                            if x in label_encoders[col].classes_ else -1)
            else:
                df[col] = df[col].apply(lambda x: label_encoders[col].transform([x])[0] 
                                        if x in label_encoders[col].classes_ else -1)
    return df, label_encoders

# Train model if not already saved
if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
    if not data.empty:
        df, encoders = preprocess_data(data, training=True)

        # Splitting data
        X = df.drop(columns=['delayed'])
        y = df['delayed']

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model and encoders
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODER_FILE, "wb") as f:
            pickle.dump(encoders, f)

        print("Model and encoders trained and saved.")
    else:
        print("Warning: Dataset is empty, skipping model training.")

# Load trained model and encoders
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        label_encoders = pickle.load(f)
else:
    model = None
    label_encoders = {}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not found. Please train the model first.", 500

    try:
        # Get input features
        features = {
            'flight_number': request.form.get('flight_number', 'unknown'),
            'scheduled_arrival': request.form.get('scheduled_arrival', 'unknown'),
            'scheduled_departure': request.form.get('scheduled_departure', 'unknown'),
            'actual_departure': request.form.get('actual_departure', 'unknown'),
            'month': request.form.get('month', 'unknown'),
            'origin': request.form.get('origin', 'unknown'),
            'destination': request.form.get('destination', 'unknown'),
            'weather': request.form.get('weather', 'normal'),
            'air_traffic_control': request.form.get('air_traffic_control', 'normal'),
            'airline_operations': request.form.get('airline_operations', 'free'),
            'airport_operations': request.form.get('airport_operations', 'free'),
            'passenger_related': request.form.get('passenger_related', 'no'),
            'external_factors': request.form.get('external_factors', 'no')
        }

        # Convert input features using label encoders
        input_data = pd.DataFrame([features])
        for col in input_data.columns:
            if col in label_encoders:
                le = label_encoders[col]
                if features[col] in le.classes_:
                    input_data[col] = le.transform([features[col]])[0]
                else:
                    input_data[col] = -1  # Assign unknown category

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_text = "Delayed" if prediction == 1 else "On Time"

        # Generate delay analysis graph
        if not data.empty:
            delay_trends = data.groupby('origin')['delayed'].sum()
            plt.figure(figsize=(8, 5))
            plt.plot(delay_trends.index, delay_trends.values, marker='o', linestyle='-', color='red')
            plt.xlabel("Airport Origin")
            plt.ylabel("Number of Delayed Flights")
            plt.title("Flight Delays by Origin (Line Chart)")
            plt.xticks(rotation=45)
            plt.grid(True)

            # Save graph dynamically
            graph_path = "static/delay_analysis.png"
            plt.savefig(graph_path, format="png", bbox_inches="tight")
            plt.close()
        else:
            print("Warning: No data available to generate the delay analysis graph.")

        return render_template('result.html', prediction=prediction_text, features=features)

    except Exception as e:
        return f"Error: {str(e)}", 500  # Handle unexpected errors gracefully

if __name__ == '__main__':
    app.run(debug=True)
