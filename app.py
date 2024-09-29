import streamlit as st
#from dotenv import load_dotenv
import os
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from PyPDF2 import PdfReader
import tempfile
import logging

# Load environment variables
#load_dotenv()
#api_key = os.getenv("GROQ_API_KEY")

#if not api_key:
#    st.error("API Key not loaded")

# Add customization options to the sidebar
st.sidebar.title('Customization')
api_key = st.sidebar.text_input("Groq API Key", type = "password")
model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
#chat_memory_token_limit = st.sidebar.number_input("Insert chat memory token limit", value = 3900, min_value = 500, max_value = 3900, placeholder="Type a token limit...")

try:
    llm = Groq(model=model, api_key = api_key)
except Exception as e:
    logging.error(f"Error initializing Groq model: {e}")
    st.error(f"Error initializing Groq model: {e}")

embed_model = HuggingFaceEmbedding(model_name = 'mixedbread-ai/mxbai-embed-large-v1')

# Set up LLM and Embedding model in settings
Settings.llm = llm
Settings.embed_model = embed_model

# Streamlit App Title
st.title("PDF Chat with Contextual Memory")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    # Read the uploaded file
    pdf_reader = PdfReader(uploaded_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Verify if the file exists and is readable
    st.write("File uploaded successfully!")

    # Initialize the chat memory
    
    memory = ChatMemoryBuffer.from_defaults(token_limit = 3900)

     # Create a temporary directory to store the text
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "temp_file.txt")
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(text)
    # Create a SimpleDocumentReader from the text
    attention_pdf = SimpleDirectoryReader(temp_dir).load_data()


    # Create the VectorStoreIndex from the loaded PDF
    index = VectorStoreIndex.from_documents(attention_pdf)
    chat_engine = CondensePlusContextChatEngine.from_defaults(index.as_retriever(), memory=memory, llm=llm)
    
    
    # User input for asking a question
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        try:
            # Query the engine
            response = chat_engine.chat(user_question)
            
            # Display the response
            st.write(f"**Response:** {response}")
            
        except Exception as e:
            logging.error(f"Error querying chat engine: {e}")
            st.error(f"Error querying chat engine: {e}")




