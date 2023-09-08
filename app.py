import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load the dataset
data_url = 'https://raw.githubusercontent.com/secluded03/Churn-Rate-Predictor/main/customer_churn_large_dataset.csv'
data = pd.read_csv(data_url)

train_data = data.copy()
train_data.drop('CustomerID', axis=1, inplace=True)
train_data.drop('Name', axis=1, inplace=True)
one_hot_encoded_data = pd.get_dummies(train_data, columns=['Gender', 'Location'])
train_data = one_hot_encoded_data

cols = ["Age","Gender_Female","Gender_Male","Location_Los Angeles","Location_Chicago","Location_New York","Location_Houston","Location_Miami","Subscription_Length_Months","Monthly_Bill","Total_Usage_GB"] 
X = train_data[cols]
Y = train_data['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.fit_transform(X_test)

# Train a logistic regression model (you can replace this with your actual training code)
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
clf = LogisticRegression(random_state=0)
clf_cv = GridSearchCV(clf, param_grid, cv = 5)
clf_cv.fit(x_train,y_train)


# Function to preprocess input data and make predictions
def predict_churn(customer_id, name, age, gender, location, subscription_length_months, monthly_bill, total_usage_gb):
    # Create a DataFrame from the input data
    input_data=pd.DataFrame({
        'Age': [age],
        'Gender_Female': [1 if gender == "Female" else 0],  # One-hot encoding for Gender
        'Gender_Male': [1 if gender == "Male" else 0],
        'Location_Los Angeles': [1 if location == "Los Angeles" else 0],
        'Location_Chicago': [1 if location == "Chicago" else 0],
        'Location_New York': [1 if location == "New York" else 0],
        'Location_Houston': [1 if location == "Houston" else 0],
        'Location_Miami': [1 if location == "Miami" else 0],  
        'Subscription_Length_Months': [subscription_length_months],
        'monthly bill': [monthly_bill],
        'Total_usage_GB': [total_usage_gb]
    })

    
# Scale the input data using the same scaler used during training
    scaled_features = scaler.transform(input_data.values)  # Use the pre-fitted scaler

    # Make predictions
    churn_rate = clf_cv.predict(scaled_features)

    return churn_rate[0]

# Streamlit UI
st.title("Churn Rate Prediction")

# Input fields
customer_id = st.text_input("Customer ID")
name = st.text_input("Name")
age = st.number_input("Age", min_value=0)
gender = st.radio("Gender", ["Male", "Female"])
location = st.text_input("Location (Country)")
subscription_length_months = st.number_input("Subscription Length (Months)", min_value=0)
monthly_bill = st.number_input("Monthly Bill",step=1.,format="%.2f")
total_usage_gb = st.number_input("Total Usage (GB)", min_value=0)

# Make prediction on button click
if st.button("Predict Churn Rate"):
    # Predict churn rate
    churn_rate_prediction = predict_churn(
        customer_id, name, age, gender, location,
        subscription_length_months, monthly_bill, total_usage_gb
    )

    st.write(f"Predicted Churn Rate: {churn_rate_prediction:.2f}")
