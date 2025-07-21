from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

def train_model(X, y, model):
    """
    Trains a machine learning model.

    Args:
        X (pd.Series or np.array): The feature set.
        y (pd.Series or np.array): The target variable.
        model: The machine learning model to train.

    Returns:
        The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.

    Args:
        model: The trained model.
        X_test (pd.Series or np.array): The test feature set.
        y_test (pd.Series or np.array): The test target variable.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, f1_score, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1_score, cm

def get_predictions(model, X):
    """
    Gets predictions from a trained model.

    Args:
        model: The trained model.
        X (pd.Series or np.array): The data to predict on.

    Returns:
        np.array: The model's predictions.
    """
    return model.predict(X)

def save_model(model, filepath):
    """
    Saves the model to a file using joblib.

    Args:
        model: The model to save.
        filepath (str): The path to save the model to.
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Loads a model from a file using joblib.

    Args:
        filepath (str): The path to the model file.

    Returns:
        The loaded model.
    """
    return joblib.load(filepath) 