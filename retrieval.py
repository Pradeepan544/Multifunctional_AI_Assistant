import chromadb
from sentence_transformers import SentenceTransformer
from llm_abstraction import generate_response
import os
import logging
import time

# Configure logging
logging.basicConfig(
    filename="retrieval.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="document_store")

def ingest_documents(text):
    """Ingests processed text into the vector database."""
    if not text.strip():
        logging.warning("Attempted to ingest an empty document.")
        return "No valid text to ingest."
    
    try:
        # Generate embeddings
        embedding = embedding_model.encode(text).tolist()
        
        # Store in ChromaDB
        doc_id = str(hash(text))  # Unique ID
        collection.add(documents=[text], embeddings=[embedding], ids=[doc_id])
        
        # ✅ Verify document count after adding
        doc_count = collection.count()
        logging.info(f"Document ingested successfully. Total documents: {doc_count}")

        return f"Document successfully ingested. Total documents: {doc_count}"
    
    except Exception as e:
        logging.error(f"Error during document ingestion: {e}")
        return "An error occurred while ingesting the document."

def retrieve_documents(query, model_name, chat_history, persona, top_k=3):
    """Retrieves relevant documents and generates a response using the selected LLM."""
    if not query.strip():
        logging.warning("Received an empty query.")
        return "Query cannot be empty."
    
    start_time = time.time()  # Track response time

    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search for similar documents
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        logging.info(f"Query received: '{query}' | Model: {model_name} | Top-{top_k} results retrieved.")
        
        if results and results["documents"] and results["documents"][0]:
            retrieved_docs = "\n".join(results["documents"][0])

            # Build conversation history into context
            history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history])

            # Persona adaptation
            persona_styles = {
                "Technical": "Provide a precise and technical explanation.",
                "Professional": "Give a clear, well-structured, and formal response.",
                "Casual": "Respond in a friendly and conversational tone."
            }
            
            prompt = f"""
            Context: {retrieved_docs}
            
            Chat History:
            {history_text}
            
            User Query: {query}
            
            Persona Style: {persona_styles.get(persona, "Professional")}
            
            Answer:
            """

            response = generate_response(prompt)
            chat_history.append({"role": "assistant", "text": response})

            logging.info(f"Response generated in {time.time() - start_time:.2f} seconds.")
            return response
        else:
            logging.warning(f"No relevant documents found for query: '{query}'")
            return "No relevant documents found."
    
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        return "An error occurred while retrieving documents."

if __name__ == "__main__":
    # Example Usage
    sample_text = "Machine learning is a method of data analysis that automates analytical model building."
    print(ingest_documents(sample_text))
    print(retrieve_documents("What is machine learning?", "gemini", [], "Professional"))
