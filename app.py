import streamlit as st
import os
from data_processing import extract_text, preprocess_text
from nlp_tasks import summarize_text, analyze_sentiment, extract_named_entities, answer_question, generate_code
from llm_abstraction import set_llm_model, get_selected_model
from retrieval import ingest_documents, retrieve_documents

def main():
    st.title("LLM-Powered NLP Application")

    # Initialize conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # LLM Model Selection
    selected_model = st.selectbox("Choose LLM Model", ["gemini", "mistral"])
    set_llm_model(selected_model)

    # Persona Selection
    persona = st.selectbox("Choose Response Style", ["Technical", "Professional", "Casual"])
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a file (CSV, JSON, TXT, PDF)", type=["csv", "json", "txt", "pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            extracted_text = extract_text(uploaded_file)
            if extracted_text:
                cleaned_text = preprocess_text(extracted_text)
                st.success("File successfully processed!")
                st.text_area("Extracted & Processed Text", cleaned_text, height=200)
                
                # ✅ Ingest text into ChromaDB  
                ingestion_result = ingest_documents(cleaned_text)
                st.write(ingestion_result)  # Show ingestion status

                # NLP Tasks
                st.subheader("Perform NLP Tasks")
                
                if st.button("Summarize Text"):
                    summary = summarize_text(cleaned_text)
                    st.write("### Summary:", summary)
                    
                if st.button("Sentiment Analysis"):
                    sentiment = analyze_sentiment(cleaned_text)
                    st.write("### Sentiment:", sentiment)
                    
                if st.button("Extract Named Entities"):
                    entities = extract_named_entities(cleaned_text)
                    st.write("### Named Entities:", entities)
                    
                st.subheader("Ask a Question")
                question = st.text_input("Enter your question:")
                if st.button("Get Answer") and question:
                    # Add question to history
                    st.session_state.chat_history.append({"role": "user", "text": question})
                    answer = answer_question(question, cleaned_text)
                    st.session_state.chat_history.append({"role": "assistant", "text": answer})
                    st.write("### Answer:", answer)

                # Display conversation history
                st.subheader("Chat History")
                for msg in st.session_state.chat_history:
                    role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
                    st.write(f"{role} {msg['text']}")

                    
                st.subheader("Code Generation")
                code_prompt = st.text_area("Enter code-related request:")
                if st.button("Generate Code") and code_prompt:
                    code = generate_code(code_prompt)
                    st.code(code, language='python')
            else:
                st.error("Could not extract text from the uploaded file.")

    # Query VectorDB
    st.subheader("Retrieve Documents")
    query = st.text_input("Enter your search query:")
    if st.button("Retrieve") and query:
        selected_model = get_selected_model() 
        if selected_model:
            retrieved_docs = retrieve_documents(query, selected_model, st.session_state.chat_history, persona)
            st.write("### Retrieved Documents:", retrieved_docs)
        else:
            st.error("Please select a model before retrieving documents.")
    
if __name__ == "__main__":
    main()
