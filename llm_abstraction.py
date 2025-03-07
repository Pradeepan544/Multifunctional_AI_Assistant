import os
from dotenv import load_dotenv
import google.generativeai as genai
from mistralai import Mistral  # Corrected import

# Load API keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Gemini & Mistral
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)  # Corrected client initialization

# Global variable for selected model
selected_model = None  

def set_llm_model(model_name):
    """Set the global LLM model selection."""
    global selected_model
    if model_name in ["gemini", "mistral"]:
        selected_model = model_name
    else:
        raise ValueError("Invalid model selected. Choose either 'gemini' or 'mistral'.")
    
def get_selected_model():
    """Returns the currently selected model."""
    return selected_model

def generate_response(prompt):
    """Calls the selected LLM model"""
    global selected_model
    if selected_model is None:
        return "No model selected. Please set the LLM model first."

    if selected_model == "gemini":
        response = gemini_model.generate_content(prompt)
        return response.text
    elif selected_model == "mistral":
        response = mistral_client.chat.complete(
            model="mistral-large-latest",  # Explicitly specify the model
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content  # Corrected response parsing

    return "Invalid model."