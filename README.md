# Multi-Functional AI Assistant

This project is a multi-functional AI assistant leveraging large language models (LLMs) to process various data formats, execute multiple NLP tasks, integrate document retrieval mechanisms, and manage user interactions. The assistant is built using Python and integrates Gemini and Mistral for LLMs, ChromaDB for vector-based document retrieval, and All-MiniLM-L6-v2 for embeddings.

## Features

- **LLM Agnostic Design**: Supports multiple LLMs, including Gemini and Mistral.
- **Data Processing**: Handles CSV, JSON, TXT, and PDF files with preprocessing capabilities.
- **NLP Functionalities**:
  - Text Summarization
  - Sentiment Analysis
  - Named Entity Recognition (NER)
  - Question Answering
  - Code Generation & Assistance
- **Retrieval-Augmented Generation (RAG)**:
  - Document ingestion and indexing using ChromaDB
  - Semantic search with vector embeddings
  - Context-aware responses using retrieved documents
- **Conversational Interface**:
  - Multi-turn dialogue with memory
  - Persona switching (Technical, Professional, Casual)
  - Interactive UI using Streamlit
- **Performance Optimization & Robustness**:
  - Caching to minimize redundant LLM calls
  - Optimized prompt engineering
  - Logging and monitoring for debugging and performance tracking
- **Testing & Evaluation**:
  - Unit tests for core functionalities
  - Integration tests for end-to-end validation

---

## Project Architecture

```
📂 AI_Assistant_Project
│── data_processing.py  # Handles data ingestion & preprocessing
│── nlp_tasks.py        # Implements core NLP functionalities
│── ── retrieval.py        # Document retrieval using ChromaDB
│── config.py           # Configuration management for LLMs & APIs
│── app.py              # Streamlit-based user interface
│── 📁 tests
│   ├── test_processing.py  # Unit tests for data processing
│   ├── test_nlp.py         # Unit tests for NLP functionalities
│   ├── test_retrieval.py   # Unit tests for retrieval mechanism
│── .env                    # Stores API keys & configurations
│── requirements.txt         # List of dependencies
│── README.md                # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
$ git clone https://github.com/your-username/AI_Assistant_Project.git
$ cd AI_Assistant_Project
```

### 2. Set Up a Virtual Environment
```bash
$ python -m venv venv
$ source venv/bin/activate   # On MacOS/Linux
$ venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
$ pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file and add the following:
```
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### 5. Run the Application
```bash
$ streamlit run ui/app.py
```

---


Feel free to contribute or reach out for collaboration! 🚀

