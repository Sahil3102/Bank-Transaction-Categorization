import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing special characters, and extra whitespace.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Performs full preprocessing on text including cleaning, tokenization, 
    stopword removal, and lemmatization.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The processed text.
    """
    text = clean_text(text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def get_vectorizer(max_features=1000):
    """
    Creates and returns a TF-IDF Vectorizer.

    Args:
        max_features (int): The maximum number of features for the vectorizer.

    Returns:
        TfidfVectorizer: An instance of TfidfVectorizer.
    """
    return TfidfVectorizer(max_features=max_features) 