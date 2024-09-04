import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
#Load the updated live modelto be made live
model = joblib.load("liveModelV1.pkl")
#Load the data to check accuracy
data = pd.read_csv('mobile_price_range_data.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Train test split for accuarcy calculation on only the testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Make predictions for x_test
y_pred = model.predict(x_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

st.title("Model Accuracy and Real-Time Prediction")

#Display Accuracy
st.write(f"Model {accuracy}")

#Real time prediction based on user inputs
st.header("Real-Time Prediction")
input_data = []
for col in x_test.columns:
    input_value = st.number_input(f'Input for feature {col}', value = 0)
    input_data.append(input_value)

#Convert input data to dataframe
input_df = pd.DataFrame([input_data], columns = x_test.columns)

#Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')