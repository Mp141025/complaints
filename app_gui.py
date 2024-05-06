

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the data
@st.cache  # Cache the data for faster loading
def load_data():
    return pd.read_csv('row2.csv')

df = load_data()

# Sidebar for user input
st.sidebar.header('Enter Your Complaint')
user_input = st.sidebar.text_input("Please enter your complaint:")

if user_input:
    # Load the model
    @st.cache  # Cache the model for faster loading
    def load_model():
        # Prepare the data
        df2 = df[['Product', 'Consumer complaint narrative']].copy()
        df2 = df2[pd.notnull(df2['Consumer complaint narrative'])]

        # Load TF-IDF Vectorizer
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
        features = tfidf.fit_transform(df2['Consumer complaint narrative']).toarray()

        # Load the classifier
        model = LinearSVC()
        model.fit(features, df2['Product'])

        return model, tfidf

    model, tfidf = load_model()

    # Get the top predicted classes
    def predict_class(user_input, model, tfidf):
        features = tfidf.transform([user_input])
        prediction = model.predict(features)
        return prediction[0]

    prediction = predict_class(user_input, model, tfidf)

    # Filter DataFrame to get documents corresponding to the predicted class
    filtered_df = df[df['Product'] == prediction]

    # Function to extract nouns from text
    def extract_nouns(text):
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        tagged = pos_tag(words)
        nouns = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
        return nouns

    # Apply the function to each document and prepare a list of unique nouns
    list_of_nouns = [noun for text in filtered_df['Consumer complaint narrative'] for noun in extract_nouns(text)]
    unique_nouns = list(set(list_of_nouns))

    st.write(f"Predicted Product: {prediction}")
    st.write("Keywords:")
    st.write(unique_nouns)

    # Generate a question based on the complaint and keywords
    st.write("Generated Question:")
    # Your code for generating a question goes here

# Run the Streamlit app
if __name__ == '__main__':
    st.title('Complaint Classifier and Question Generator')
    st.write("Welcome to the Complaint Classifier and Question Generator app!")
