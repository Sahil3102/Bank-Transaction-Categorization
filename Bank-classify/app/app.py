import streamlit as st
import pandas as pd
import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_text
from src.model import load_model

st.set_page_config(page_title="Bank Transaction Categorization", layout="wide")

st.title("Bank Transaction Categorization")

st.write("""
Upload a CSV file with 'Description' and 'Amount' columns to get the predicted categories for your bank transactions.
""")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error("Model not found. Please run the `notebooks/EDA_and_Model.ipynb` notebook to train and save the model.")
    st.stop()

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if 'Description' in data.columns:
        # Preprocess the data
        data['Processed_Description'] = data['Description'].apply(preprocess_text)
        
        # Get predictions
        predictions = model.predict(data['Processed_Description'])
        data['Predicted_Category'] = predictions
        
        st.subheader("Categorized Transactions")
        st.dataframe(data[['Description', 'Amount', 'Predicted_Category']])
        
        # Display category distribution
        st.subheader("Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='Predicted_Category', data=data, order=data['Predicted_Category'].value_counts().index, ax=ax)
        ax.set_title('Distribution of Predicted Categories')
        ax.set_xlabel('Count')
        ax.set_ylabel('Category')
        st.pyplot(fig)
        
    else:
        st.error("The uploaded CSV file must contain a 'Description' column.") 