import pandas as pd
import json
import re
import spacy
import asyncio
import pypdf  # Import pypdf (formerly PyPDF2)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Fix for asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nltk.download("stopwords")
nltk.download("wordnet")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def extract_text(file):
    print(f"Processing file: {file.type}")  # Debugging output

    if file.type == "text/plain":  # TXT
        return file.getvalue().decode("utf-8")
    elif file.type == "application/json":  # JSON
        data = json.load(file)
        return json.dumps(data, indent=4)
    elif file.type == "text/csv":  # CSV
        df = pd.read_csv(file)
        return df.to_string()
    elif file.type == "application/pdf":  # PDF
        text = ""
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            '''print(f"Extracted text from page: {extracted}")  # Debugging output'''
            
            if extracted:
                text += extracted + "\n"

        if not text.strip():
            print("PDF text extraction failed or empty.")  # Debugging output

        return text.strip()

    print("Unsupported file type.")  # Debugging output
    return None  # If file type is unsupported

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters & numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenization using spaCy instead of NLTK
    doc = nlp(text)
    
    # Stopword removal & lemmatization
    processed_tokens = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words]
    
    # Reconstruct cleaned text
    return " ".join(processed_tokens)