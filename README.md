# Fake News Detection Web App

This project is a web application built using Streamlit to detect fake news articles by applying Machine Learning techniques. It incorporates various Data Science principles such as **data analytics**, **data mining**, **data cleaning**, **feature exploration**, **prediction modeling**, and **data visualization** to distinguish between real and fake news.

## Principles of Data Science Applied

### 1. **Data Analytics**

- Analyzing the datasets of real and fake news articles to discover patterns and gain insights.
- Applying descriptive statistics to understand word frequency, article length, and the distribution of real vs fake news.

### 2. **Data Mining**

- Mining patterns from the dataset using Natural Language Processing (NLP) techniques.
- Extracting and processing textual features such as keywords and common phrases in fake vs real news.

### 3. **Data Cleaning**

- Preprocessing textual data by:
  - Converting text to lowercase.
  - Removing special characters and numbers.
- Ensures the model is trained on clean data, enhancing the prediction accuracy.

### 4. **Feature Exploration**

- Using **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency) to transform the text into numerical features.
- Selecting the top 5000 most important words as features to help the model classify news articles effectively.

### 5. **Prediction Modeling**

- A **Multinomial Naive Bayes** classifier is trained on the data to predict whether a news article is real or fake.
- The data is split into training and testing sets, with 80% used for training and 20% for testing the model's performance.

### 6. **Data Visualization**

- The web app interface provides real-time predictions and visual insights into the dataset.
- Future work could include visualizing the most common words in both real and fake news and creating charts to show class distribution.

## Tech Stack

- **Python**: Backend logic and machine learning model.
- **Streamlit**: For building an interactive web application.
- **Scikit-learn**: Machine learning and text vectorization.
- **Pandas**: Data manipulation and analysis.
- **Joblib**: For saving and loading the trained model.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```
# Fake_News_Detection
