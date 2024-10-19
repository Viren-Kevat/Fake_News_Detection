# 📰 Fake News Detection Web App

Welcome to the **Fake News Detection Web App**, a dynamic platform built using **Streamlit** to tackle the challenge of misinformation. By leveraging cutting-edge **Machine Learning** techniques, this application helps users differentiate between real and fake news articles with ease. 

## 🌟 Key Data Science Principles Applied

### 1. **Data Analytics**
- Dive deep into the datasets of real and fake news articles to uncover **patterns** and **insights**.
- Utilize descriptive statistics to analyze **word frequency**, **article length**, and the distribution between real and fake news.

### 2. **Data Mining**
- Harness the power of **Natural Language Processing (NLP)** techniques to extract valuable insights from the dataset.
- Identify and process **textual features** such as keywords and common phrases that characterize fake and real news.

### 3. **Data Cleaning**
- **Preprocess** textual data by:
  - Converting text to lowercase.
  - Removing special characters and numbers.
- This step ensures the model is trained on **clean data**, significantly enhancing prediction accuracy.

### 4. **Feature Exploration**
- Employ **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency) to transform text into numerical features.
- Select the **top 5000** most impactful words as features to empower the model in classifying news articles effectively.

### 5. **Prediction Modeling**
- Train a **Multinomial Naive Bayes** classifier on the data to predict whether a news article is real or fake.
- Split the data into training (80%) and testing (20%) sets to evaluate the model's performance accurately.

### 6. **Data Visualization**
- The app interface provides **real-time predictions** along with insightful visualizations of the dataset.
- Future enhancements could include visualizing the **most common words** in both real and fake news, as well as creating charts to illustrate class distributions.

## ⚙️ Tech Stack
- **Python**: Core logic and machine learning model implementation.
- **Streamlit**: Framework for building an interactive web application.
- **Scikit-learn**: Essential for machine learning and text vectorization.
- **Pandas**: Powerful data manipulation and analysis library.
- **Joblib**: Efficient saving and loading of the trained model.

## 🚀 Setup Instructions

### 1. Clone the Repository
To get started, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```
### 2. Prepare Your CSV Files
**Important**: Ensure you have the following CSV files in the project directory:

- **True.csv**: Contains real news articles with a column named `text`.
- **Fake.csv**: Contains fake news articles with the same structure.

### 3. File Format
**Consistency Matters**: Both CSV files must have a column labeled `text` containing the news titles and articles.

**Avoid Errors**: Structured data prevents processing errors.

### 4. Install Dependencies
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
### 5. Run the Application
To start the Streamlit application, run:
```bash
streamlit run app.py
```
### 6. Usage
Once the app is running, you can train the model by clicking the "Train Model" button.  
Enter a news article or headline in the text area to check if it's Fake or Real, and click the "Predict" button.

### Acknowledgments
This project leverages the power of Machine Learning and Natural Language Processing to combat the spread of misinformation.

**Feel free to adjust any sections according to your preferences or additional features!**
