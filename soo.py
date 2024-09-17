import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
file_path = r'C:\Users\premi\Desktop\Premila\projects\Singapore\Resale_flat.csv' 
df = pd.read_csv(file_path)

# Helper function to parse 'remaining_lease'
def parse_remaining_lease(value):
    try:
        years, months = value.split(' years ')
        months = months.replace(' months', '')
        return int(years) * 12 + int(months)  # Total months
    except:
        return None

# Preprocess the data
def preprocess_data(df, is_training=True):
    # Categorical and numerical columns
    categorical_columns = ['town','flat_type','block','street_name','storey_range','flat_model']
    numerical_columns = ['floor_area_sqm','lease_commence_date']
    
    # Fill missing values
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    # Convert 'remaining_lease' to numeric
    df['remaining_lease'] = df['remaining_lease'].apply(parse_remaining_lease)
    df['remaining_lease'] = pd.to_numeric(df['remaining_lease'], errors='coerce')
    df['remaining_lease'].fillna(df['remaining_lease'].mean(), inplace=True)

    # Extract and one-hot encode 'month' if needed
    if is_training and 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], errors='coerce').dt.month
        one_hot = pd.get_dummies(df['month'], prefix='month')
        df = pd.concat([df, one_hot], axis=1)
        df.drop(['month'], axis=1, inplace=True)

    # Feature engineering
    df['lease_age'] = 2024 - pd.to_datetime(df['lease_commence_date'], errors='coerce').dt.year

    # Convert 'storey_range' to numerical
    df['storey_range'] = df['storey_range'].apply(lambda x: np.mean([int(i) for i in x.split(' TO ')]))

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Define features and target variable for training
    if is_training:
        X = df.drop(columns=['resale_price'])
        y = df['resale_price']
        
        # Scaling numerical features
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        
        return X, y, scaler  # Return the scaler for future use
    else:
        # Transform the input data during prediction
        X = df.copy()
        X[numerical_columns] = StandardScaler().fit_transform(X[numerical_columns])
        return X

# Train the model
def model_train():
    # Preprocess training data
    X, y, scaler = preprocess_data(df)

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Streamlit App
st.title("Singapore HDB Resale Price Prediction")

# Check if the model is trained
if 'model' not in st.session_state:
    st.write("Training the model, please wait...")
    model, scaler = model_train()
    st.session_state.model = model
    st.session_state.scaler = scaler
else:
    model = st.session_state.model
    scaler = st.session_state.scaler

# User input for prediction
town = st.selectbox('Select Town', ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
flat_type = st.selectbox('Select Flat Type', ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM', 'MULTI-GENERATION'])
flat_model = st.selectbox('Select Flat Model', ['Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment', 'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat', 'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2', 'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette', 'Multi Generation', 'Premium Apartment Loft', '2-room', '3Gen'])
floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0, max_value=200, step=1)
storey_range = st.number_input('Storey Range', min_value=1, max_value=50, step=1)
lease_age = st.number_input('Lease Age (Years)', min_value=1, max_value=99, step=1)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'town': [town],
    'flat_type': [flat_type],
    'flat_model': [flat_model],
    'floor_area_sqm': [floor_area_sqm],
    'storey_range': [storey_range],
    'lease_age': [lease_age],
})

# Apply the same preprocessing as during training
input_data = preprocess_data(input_data, is_training=False)

# Predict resale price
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Estimated Resale Price: <span style='font-size: 1.5em; color: green;'>${prediction[0]:,.2f}</span>", unsafe_allow_html=True)
Ss