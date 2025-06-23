# 📄 PDF Chatbot 🧠

### A lightweight Python-based chatbot that lets you query PDFs using a retrieval-augmented generation (RAG) approach powered by LangChain, FAISS, and any LLM on Huggingface. No web UI dependencies—ideal for CLI use or easy integration into your own frontend.

### 🔍 Key Features

  - PDF Loader & Parser: Extracts text from uploaded PDF files.
  - Chunking: Splits content into optimal chunks (e.g., 500–1000 tokens).
  - Embedding & Vector Store: Uses FAISS vector store.
  - Retriever + LLM: Retrieves top-k relevant chunks, sends them plus the prompt to an LLM.
  - Flexible Interface: Designed for CLI usage; can hook into a simple web/chat UI later.

### 🚀 Quick Start

  1. Install dependencies
       pip install -r requirements.txt
  
  2. Add your PDFs
       Place all PDF files you want to use into the pdf_store/ directory.
  
  3. Create a .env file
      Include your Hugging Face token (or other credentials):
      HF_TOKEN=<your_huggingface_token_here>
  
  4. Generate embeddings (run once or whenever PDFs update)
      python embedding.py
  
  5. Start the chat interface
      python chat.py

### 🏗️ Architecture Diagram
  ![Your paragraph text](https://github.com/user-attachments/assets/c90b7e04-fdc9-4be5-9840-f96bc8280f6d)
  
### ✅ Future Enhancements

 - Add docx/txt loader support.
 - Expose optional Streamlit or Gradio web chat front-end.
 - Implement conversation memory and session history.
 - Integrate multiple PDF support with merged retrieval.
 - Introduce error handling for OCR or parsing failures.
