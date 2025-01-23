import streamlit as st
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('house_data.csv')

# Load data
df = load_data()

# Streamlit app
st.title("House Price Prediction App")

# Display the dataset
st.write("### Dataset Preview")
st.write(df)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Train the model
@st.cache_data
def train_model(data):
    # Split the data into features (X) and target (y)
    X = data[['SquareFeet', 'NumRooms']]
    y = data['Price']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    return model, mae

# Train the model and get MAE
model, mae = train_model(df)

st.write("### Model Training")
st.write(f"Model trained successfully! Mean Absolute Error: ${mae:,.2f}")


# User Input Section
st.sidebar.header("Input Features")
square_feet = st.sidebar.slider("Square Feet", int(df['SquareFeet'].min()), int(df['SquareFeet'].max()), step=50)
num_rooms = st.sidebar.slider("Number of Rooms", int(df['NumRooms'].min()), int(df['NumRooms'].max()), step=1)

# Make prediction
input_features = [[square_feet, num_rooms]]
predicted_price = model.predict(input_features)[0]

st.write("### Predicted House Price")
st.write(f"The estimated price for a house with **{square_feet} square feet** and **{num_rooms} rooms** is:")
st.write(f"### ${predicted_price:,.2f}")

