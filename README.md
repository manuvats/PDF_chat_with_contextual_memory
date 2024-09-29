# PDF_chat_with_contextual_memory
This is an AI application using Groq API and LlamaIndex (an LLM framework). This application loads the text from a PDF file, converts it into embeddings, and saves it into the vector store. After that, it converts the vector store into the retriever that is used to build a RAG chat engine with history. 

In short, we have built a context-aware ChatPDF application to help you understand the document much faster.

## How to run the project
After cloning the repository, type the following command in your terminal-:

streamlit run app.py --server.enableXsrfProtection false
