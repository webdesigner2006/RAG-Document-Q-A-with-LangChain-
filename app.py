import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

# --- Constants ---
# Using instructor-base is a good trade-off between performance and size.
EMBEDDING_MODEL_NAME = "hkunlp/instructor-base" 
# Mistral-7B is a powerful open-source model.
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
TEMP_DOCS_DIR = "documents"

# --- App Configuration ---
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")
st.header("📄 Chat with Your Document")

# --- API Key Management ---
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_token:
    st.warning("Hugging Face API Token not found in .env file.")
    hf_api_token = st.text_input(
        "Please enter your Hugging Face API Token:", type="password", key="api_token_input"
    )
    if hf_api_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
    else:
        st.stop()

# --- Caching Functions for Performance ---
@st.cache_resource
def load_embeddings():
    """Loads the sentence transformer model for embeddings."""
    return HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm():
    """Loads the language model from Hugging Face Hub."""
    return HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 1024}
    )

# This function is now cached to avoid reprocessing the same file.
# The key is the file's content, so it re-runs if a new file is uploaded.
@st.cache_data(show_spinner="Analyzing document...")
def process_pdf(_uploaded_file):
    """Loads, splits, and embeds the PDF, creating a FAISS vector store."""
    if _uploaded_file is not None:
        # Create a temporary directory to store the file
        os.makedirs(TEMP_DOCS_DIR, exist_ok=True)
        temp_file_path = os.path.join(TEMP_DOCS_DIR, _uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(_uploaded_file.getbuffer())

        # 1. Load the document
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 2. Split the doc into smaller chunks.
        # This is crucial for the LLM to find relevant context.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings and the vector store
        embeddings_model = load_embeddings()
        vector_store = FAISS.from_documents(texts, embeddings_model)

        os.remove(temp_file_path) # Clean up the temp file
        
        return vector_store
    return None

# --- Streamlit UI ---
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about a PDF you upload. "
    "It finds the most relevant text chunks from the document and uses an LLM to generate an answer.\n\n"
    "**Tech Stack:**\n- Streamlit\n- LangChain\n- Hugging Face\n- FAISS"
)
# TODO: Add support for more document types like .txt and .docx
st.sidebar.markdown("---")


uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    vector_store = process_pdf(uploaded_file)
    if vector_store:
        st.success(f"Document '{uploaded_file.name}' is ready for questions!")
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff", # "stuff" is a simple chain type, good for smaller docs
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}) # find top 3 relevant chunks
        )

        question = st.text_input(
            "Ask a question about the document:",
            placeholder="What is the main conclusion of this paper?"
        )

        if st.button("Get Answer"):
            if question:
                print(f"User question: {question}") # A print statement for debugging
                with st.spinner("Searching for the answer..."):
                    try:
                        result = rag_chain.invoke({"query": question})
                        st.subheader("Answer:")
                        st.write(result['result'])
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
            else:
                st.warning("Please ask a question first!")
else:
    st.info("Upload a PDF to get started.")
