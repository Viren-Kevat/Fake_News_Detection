# Import libraries
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Set paths for true and fake news CSV files
TRUE_NEWS_FILE = "E:/Fake_News_Detection/True.csv"  # Replace with your actual path
FAKE_NEWS_FILE = "E:/Fake_News_Detection/Fake.csv"  # Replace with your actual path

# Load datasets (True news and Fake news)
@st.cache_data
def load_data():
    true_news = pd.read_csv(TRUE_NEWS_FILE)
    false_news = pd.read_csv(FAKE_NEWS_FILE)

    # Assign labels: 1 for real news, 0 for fake news
    true_news['label'] = 1
    false_news['label'] = 0

    # Concatenate the datasets
    df = pd.concat([true_news, false_news], ignore_index=True)
    return df[['text', 'label']]  # Ensure columns 'text' and 'label' are present

# Preprocess Text Data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters and numbers
    return text

# Prepare the model
def train_model(df):
    df['text'] = df['text'].apply(preprocess_text)

    # Split data into training and testing
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use CountVectorizer instead of TfidfVectorizer
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Test model accuracy
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and vectorizer
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return accuracy

# Predict using the trained model
def predict_fake_news(input_text):
    # Load the saved model and vectorizer
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Preprocess and vectorize input
    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text])

    # Predict and return result
    prediction = model.predict(input_vector)
    return prediction[0]

# Streamlit App
def main():
    st.title("Fake News Detection")

    # Load data from CSV files
    df = load_data()
    st.write("Data Loaded Successfully. Total articles: ", df.shape[0])

    # Option to train the model
    if st.button("Train Model"):
        accuracy = train_model(df)
        st.write(f"Model trained with an accuracy of: {accuracy * 100:.2f}%")

    # Input for user to enter news article or headline
    input_text = st.text_area("Enter news article or headline to check if it's Fake or Real:")

    # Prediction button
    if st.button("Predict"):
        if input_text:
            prediction = predict_fake_news(input_text)
            if prediction == 1:
                st.write("The news is Real.")
            else:
                st.write("The news is Fake.")

# Correct use of name
if __name__ == '__main__':
    main()
