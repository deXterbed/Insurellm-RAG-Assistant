"""
Insurellm RAG Chatbot - A Retrieval-Augmented Generation application
Built with LangChain, Chroma, and Gradio

This app allows users to ask questions about Insurellm company, 
employees, products, and contracts using semantic search.
"""

import os
import glob
import logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
KNOWLEDGE_BASE_DIR = "knowledge-base"

# Load environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")


def build_vector_store():
    """Build the vector store from knowledge base documents."""
    logger.info("Building vector store from knowledge base...")
    
    # Check if knowledge base exists
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        raise FileNotFoundError(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found!")
    
    # Load documents
    folders = glob.glob(f"{KNOWLEDGE_BASE_DIR}/*")
    text_loader_kwargs = {"encoding": "utf-8"}
    
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs=text_loader_kwargs
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    
    if not documents:
        raise ValueError(f"No documents found in {KNOWLEDGE_BASE_DIR}!")
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} document chunks")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    
    # Delete existing collection if it exists
    if os.path.exists(DB_NAME):
        try:
            Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
            logger.info("Deleted existing vector store")
        except Exception as e:
            logger.warning(f"Could not delete existing collection: {e}")
    
    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    
    logger.info(f"Vector store created successfully at {DB_NAME}")
    return vectorstore


def initialize_app():
    """Initialize the application - build vector store if needed."""
    embeddings = OpenAIEmbeddings()
    
    # Check if vector store exists
    if not os.path.exists(DB_NAME) or not os.path.exists(f"{DB_NAME}/chroma.sqlite3"):
        logger.info("Vector store not found. Building it now...")
        build_vector_store()
        logger.info("Vector store built successfully!")
    else:
        logger.info("Using existing vector store")
    
    # Load vector store
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create LLM
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    return retriever, conversation_chain


# Initialize components
logger.info("Initializing application...")
try:
    retriever, conversation_chain = initialize_app()
    logger.info("Application initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise


def chat(message, history):
    """Handle chat messages and return responses."""
    if not message or not message.strip():
        return "Please enter a question."
    
    try:
        # Log the user's question
        logger.info(f"User question: {message}")
        
        # Retrieve and log relevant documents
        try:
            retrieved_docs = retriever.get_relevant_documents(message)
        except AttributeError:
            # Fallback for newer LangChain versions
            retrieved_docs = retriever.invoke(message)
        logger.info(f"Retrieved {len(retrieved_docs)} document chunks")
        
        # Get response from chain
        response = conversation_chain.invoke({"question": message})
        answer = response["answer"]
        
        logger.info("Response generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"I encountered an error: {str(e)}. Please try again."


def reset_memory():
    """Reset the conversation memory."""
    global conversation_chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.7, model_name=MODEL),
        retriever=retriever,
        memory=memory
    )
    logger.info("Conversation memory reset")
    return "Conversation memory has been reset!"


# Create Gradio interface
with gr.Blocks(title="Insurellm RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Insurellm RAG Chatbot
        
        Ask questions about Insurellm company, employees, products, and contracts!
        
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions
        based on the knowledge base documents.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat,
                title="Chat with Insurellm Assistant",
                description="Type your question and press Enter",
                examples=[
                    "Who is Avery Lancaster?",
                    "What is Carllm?",
                    "Tell me about Insurellm",
                    "What contracts does Insurellm have?",
                    "Who are the employees?",
                ],
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            reset_btn = gr.Button("🔄 Reset Memory", variant="secondary")
            reset_output = gr.Textbox(label="Status", interactive=False)
            reset_btn.click(fn=reset_memory, outputs=reset_output)
            
            gr.Markdown(
                """
                ### About
                This chatbot uses:
                - **LangChain** for RAG pipeline
                - **Chroma** for vector storage
                - **OpenAI GPT-4o-mini** for responses
                - **Conversational memory** for context
                """
            )
    
    gr.Markdown(
        """
        ---
        **Note**: The conversation history is maintained during the session. 
        Use "Reset Memory" to start a fresh conversation.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
