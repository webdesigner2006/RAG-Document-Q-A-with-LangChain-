import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# --- App Configuration ---
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")
st.title("📄 Chat with Your Document using RAG")

# --- Hugging Face API Token ---
# It's recommended to set this as an environment variable for security
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    try:
        # A more secure way to get the token if running locally
        from dotenv import load_dotenv
        load_dotenv()
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    except ImportError:
        pass # If dotenv is not installed, we'll rely on the text input

if not HUGGINGFACEHUB_API_TOKEN:
    st.warning("Hugging Face API Token not found. Please enter it below.")
    HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your Hugging Face API Token:", type="password")

if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
else:
    st.stop()


# --- Caching Functions for Performance ---
@st.cache_resource
def load_embeddings():
    """Loads the sentence transformer model for embeddings."""
    # Using a smaller, faster model for demonstration
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

@st.cache_resource
def load_llm():
    """Loads the language model from Hugging Face Hub."""
    # Example model: Mistral-7B Instruct
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )

@st.cache_data(show_spinner="Processing PDF...")
def process_pdf(uploaded_file):
    """Loads and processes the PDF, creating a vector store."""
    if uploaded_file is not None:
        # Save the uploaded file temporarily to be read by PyPDFLoader
        temp_file_path = os.path.join("documents", uploaded_file.name)
        os.makedirs("documents", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 1. Load the document
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings and vector store
        embeddings = load_embeddings()
        vector_store = FAISS.from_documents(texts, embeddings)

        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return vector_store
    return None

# --- Streamlit UI ---
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Make sure you have a Hugging Face API Token.\n"
    "2. Upload a PDF document.\n"
    "3. Wait for the document to be processed.\n"
    "4. Ask a question about the document's content."
)

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    vector_store = process_pdf(uploaded_file)
    if vector_store:
        st.success(f"Document '{uploaded_file.name}' processed successfully!")
        
        # Initialize the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2})
        )

        # Chat interface
        question = st.text_input("Ask a question about the document:", placeholder="What is the main topic of this document?")

        if st.button("Get Answer"):
            if question:
                with st.spinner("Finding the answer..."):
                    try:
                        result = qa_chain.invoke({"query": question})
                        st.subheader("Answer:")
                        st.write(result['result'])
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF to begin.")
