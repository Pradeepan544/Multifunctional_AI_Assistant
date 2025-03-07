import spacy
from llm_abstraction import generate_response

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def summarize_text(text):
    """Summarizes text using LLM"""
    return generate_response(f"Summarize the following:\n{text}")

def analyze_sentiment(text):
    """Performs sentiment analysis using LLM"""
    return generate_response(f"Analyze the sentiment of this text:\n{text}")

def extract_named_entities(text):
    """Extracts named entities using spaCy"""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def answer_question(question, context):
    """Answers a question using LLM"""
    return generate_response(f"Context:\n{context}\n\nQuestion: {question}. Give ANSWER for the following question")

def generate_code(prompt):
    """Generates code using LLM"""
    return generate_response(f"Generate Python code for:\n{prompt}")
