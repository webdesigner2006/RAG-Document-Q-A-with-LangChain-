# RAG-Document-Q-A-with-LangChain-
This application allows you to "chat" with your documents. You upload a PDF file, and the app uses a Retrieval-Augmented Generation (RAG) pipeline with LangChain and Hugging Face models to answer questions based on the document's content.
# 📄 RAG Document Q&A

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/) Ever wished you could just ask your documents questions instead of reading through them? This app does exactly that! Upload a PDF, and our AI assistant will use it as its brain to answer any question you have about the content.

It's perfect for quickly understanding research papers, legal documents, or long reports without the headache.



---

## ✨ Features

-   **Upload & Chat**: Simply upload any PDF document.
-   **AI-Powered Answers**: Uses a modern RAG (Retrieval-Augmented Generation) pipeline to find the most relevant information and generate human-like answers.
-   **Fast & Efficient**: Caches models and processed data to give you quick responses on subsequent questions.
-   **Secure**: Your documents are processed temporarily and are not stored long-term.

---

## 🛠️ Tech Stack

-   **Language**: Python
-   **Framework**: Streamlit (for the beautiful web UI)
-   **Core AI**: LangChain (for orchestrating the RAG pipeline)
-   **LLM & Embeddings**: Hugging Face (via `HuggingFaceHub` and `sentence-transformers`)
-   **Vector Store**: FAISS (for lightning-fast information retrieval)

---

## 🚀 How to Run It Locally

Ready to chat with your own documents? Here’s how to get this running on your machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/rag_document_qa.git](https://github.com/your-username/rag_document_qa.git)
    cd rag_document_qa
    ```

2.  **Set up a Python environment:**
    It's a good practice to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your Hugging Face API Token:**
    You'll need a free Hugging Face API token to use the language models.
    -   Create a file named `.env` in the project's root directory.
    -   Add your token to this file like so:
        ```
        HUGGINGFACEHUB_API_TOKEN='hf_your_token_here'
        ```

5.  **Launch the app!**
    ```bash
    streamlit run app.py
    ```

Your browser should open with the app running. Enjoy!
