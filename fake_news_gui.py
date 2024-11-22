# Import necessary libraries
import pickle  # For loading pre-trained models and vectorizers
import streamlit as st  # For building the web app interface
import re  # For text preprocessing using regular expressions
from nltk.corpus import stopwords  # For removing common stopwords
from nltk.stem.porter import PorterStemmer  # For stemming words to their root form
import nltk  # For downloading necessary NLTK resources

# Ensure stopwords are downloaded (required for first-time execution)
nltk.download('stopwords')

# Initialize the PorterStemmer for stemming
port_stem = PorterStemmer()

# Function to preprocess the input text
def preprocess_text(content):
    """
    Cleans and preprocesses the input text for the classifier.
    Steps:
        - Remove non-alphabetic characters.
        - Convert text to lowercase.
        - Tokenize text into words.
        - Remove stopwords and apply stemming to each word.
    Args:
        content (str): Raw input text from the user.
    Returns:
        str: Preprocessed and cleaned text.
    """
    # Remove all characters except alphabets
    content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert text to lowercase
    content = content.lower()
    # Split text into individual words
    content = content.split()
    # Remove stopwords and apply stemming
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    # Join the cleaned words into a single string
    content = ' '.join(content)
    return content

# Load the pre-trained vectorizer and classification model
try:
    # Load the TF-IDF vectorizer
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    # Load the classification model
    model = pickle.load(open('model.pkl', 'rb'))
    # Display a success message if files load correctly
    st.success("Model and Vectorizer loaded successfully!")
except Exception as e:
    # Display an error message if there's an issue with loading files
    st.error(f"Error loading model or vectorizer: {e}")

# Build the Streamlit web app interface
st.title("Fake News Classifier")  # Set the app title

# Input text box for user to enter their message/news
input_sms = st.text_input("Enter the message")

# Classify button to trigger the prediction
if st.button("Classify"):
    if input_sms.strip() != "":  # Ensure input is not empty
        # Preprocess the input text
        transformed_sms = preprocess_text(input_sms)

        try:
            # Convert the preprocessed text into a vector using the TF-IDF vectorizer
            vector_input = tfidf.transform([transformed_sms])
            
            # Use the model to predict whether the news is fake or real
            result = model.predict(vector_input)[0]

            # Display the result to the user
            if result == 1:
                st.header("Fake News")
            else:
                st.header("Real News")
        except Exception as e:
            # Handle any errors during classification
            st.error(f"Error during classification: {e}")
    else:
        # Display a warning if no input was provided
        st.warning("Please enter a message to classify!")
