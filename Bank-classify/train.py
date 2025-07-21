import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import preprocess_text, get_vectorizer
from model import train_model, evaluate_model, save_model

def main():
    """
    Main function to train and save the model.
    """
    print("Loading data...")
    df = pd.read_csv('data/transactions.csv')

    print("Preprocessing data...")
    df['Processed_Description'] = df['Description'].apply(preprocess_text)

    X = df['Processed_Description']
    y = df['Category']

    # --- Train and Evaluate Logistic Regression ---
    print("Training Logistic Regression model...")
    lr_pipeline = Pipeline([
        ('vectorizer', get_vectorizer()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_model, X_test_lr, y_test_lr = train_model(X, y, lr_pipeline)
    lr_accuracy, _, _, lr_f1, _ = evaluate_model(lr_model, X_test_lr, y_test_lr)
    print(f"Logistic Regression F1-Score: {lr_f1:.4f}")

    # --- Train and Evaluate Random Forest ---
    print("Training Random Forest model...")
    rf_pipeline = Pipeline([
        ('vectorizer', get_vectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    rf_model, X_test_rf, y_test_rf = train_model(X, y, rf_pipeline)
    rf_accuracy, _, _, rf_f1, _ = evaluate_model(rf_model, X_test_rf, y_test_rf)
    print(f"Random Forest F1-Score: {rf_f1:.4f}")

    # --- Save the best model ---
    if lr_f1 > rf_f1:
        best_model = lr_model
        print("Logistic Regression is the best model.")
    else:
        best_model = rf_model
        print("Random Forest is the best model.")
        
    save_model(best_model, 'app/model.joblib')
    print(f"Best model saved to app/model.joblib")

if __name__ == "__main__":
    main() 