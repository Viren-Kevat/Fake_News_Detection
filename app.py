import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load datasets from uploaded CSV files
@st.cache_data
def load_data(true_news_file, fake_news_file):
    true_news = pd.read_csv(true_news_file)
    false_news = pd.read_csv(fake_news_file)

    true_news['label'] = 1  # 1 for real news
    false_news['label'] = 0  # 0 for fake news

    df = pd.concat([true_news, false_news], ignore_index=True)
    return df[['text', 'label']]

# Preprocess Text Data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    return text

# Train the model
def train_model(df):
    df['text'] = df['text'].apply(preprocess_text)
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and vectorizer
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return accuracy

# Predict using the trained model
def predict_fake_news(input_text):
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text])

    prediction = model.predict(input_vector)
    return prediction[0]

# Streamlit App
def main():
    st.title("Fake News Detection")

    true_news_file = st.file_uploader("Upload True News CSV", type=["csv"])
    fake_news_file = st.file_uploader("Upload Fake News CSV", type=["csv"])

    if true_news_file and fake_news_file:
        df = load_data(true_news_file, fake_news_file)
        st.write("Data Loaded Successfully. Total articles: ", df.shape[0])

        if st.button("Train Model"):
            accuracy = train_model(df)
            st.write(f"Model trained with an accuracy of: {accuracy * 100:.2f}%")

        input_text = st.text_area("Enter news article or headline to check if it's Fake or Real:")

        if st.button("Predict"):
            if input_text:
                prediction = predict_fake_news(input_text)
                if prediction == 1:
                    st.write("The news is Real.")
                else:
                    st.write("The news is Fake.")

if __name__ == '__main__':
    main()
