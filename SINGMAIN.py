from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = r"C:\Users\premi\projects\Singapore\Resale_flat.csv"
df = pd.read_csv(file_path)

# Function to convert storey range
def convert_storey_range(storey_range):
    if isinstance(storey_range, str):
        try:
            return np.mean([int(i) for i in storey_range.split(' TO ')])
        except ValueError:
            return np.nan
    return storey_range

# Preprocess the data
def preprocess_data(df, is_training=True, scaler=None, encoder=None):
    if 'lease_commence_date' in df.columns:
        df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], errors='coerce')
        df['lease_age'] = 2024 - df['lease_commence_date'].dt.year
        df.drop(columns=['lease_commence_date'], inplace=True)

    df['storey_range'] = df['storey_range'].apply(convert_storey_range)

    categorical_columns = ['town', 'flat_type', 'flat_model']
    numerical_columns = ['floor_area_sqm', 'storey_range', 'lease_age']

    if is_training:
        if 'resale_price' not in df.columns:
            raise ValueError("The 'resale_price' column is missing from the dataset.")

        y = df['resale_price']
        df.drop(columns=['resale_price', 'month', 'block', 'street_name'], inplace=True)

        # Create preprocessor
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown='ignore')
        
        # Fit and transform
        X_numerical = scaler.fit_transform(df[numerical_columns])
        X_categorical = encoder.fit_transform(df[categorical_columns]).toarray()
        
        # Combine the processed features
        X = np.hstack((X_numerical, X_categorical))
        
        return X, y, scaler, encoder

    # Transform the data using the provided scaler and encoder
    X_numerical = scaler.transform(df[numerical_columns])
    X_categorical = encoder.transform(df[categorical_columns]).toarray()
    X = np.hstack((X_numerical, X_categorical))
    return X

# Train the model
def model_train(df):
    X, y, scaler, encoder = preprocess_data(df, is_training=True)

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, encoder

# Streamlit App
st.title("Singapore HDB Resale Price Prediction")

# Train the model if not already trained
if 'model' not in st.session_state:
    st.write("Training the model, please wait...")
    model, scaler, encoder = model_train(df)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.encoder = encoder
else:
    model = st.session_state.model
    scaler = st.session_state.scaler
    encoder = st.session_state.encoder

# User input for prediction
town = st.selectbox('Select Town', sorted(df['town'].unique()))
flat_type = st.selectbox('Select Flat Type', sorted(df['flat_type'].unique()))
flat_model = st.selectbox('Select Flat Model', sorted(df['flat_model'].unique()))
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

# Apply the same preprocessing for predictions
try:
    input_data_preprocessed = preprocess_data(input_data, scaler=scaler, encoder=encoder, is_training=False,)
except Exception as e:
    st.error(f"An error occurred during input processing: {e}")

# Predict resale price
if st.button("Predict"):
    prediction = model.predict(input_data_preprocessed)
    st.write(f"Estimated Resale Price: <span style='font-size: 1.5em; color: green;'>${prediction[0]:,.2f}</span>", unsafe_allow_html=True)
