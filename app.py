from flask import Flask, render_template, jsonify, request
import os
import getpass
from src.utils import PDFProcessor
from langchain_community.llms import CTransformers
from src.logger import logging as log
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from src.prompt import prompt_template
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

app = Flask(__name__)

# Configuration
class Config:
    # Try to get from environment, fallback to getpass if not found
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key (recommend to set in .env): ")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or getpass.getpass("Groq API Key (recommend to set in .env): ")
    INDEX_NAME = "medical-chatbot"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.8
    SEARCH_CONFIG = {"k": 1, "score_threshold": 0.5}

app.config.from_object(Config)

# Error handling for missing credentials
if not all([app.config["PINECONE_API_KEY"], app.config["GROQ_API_KEY"]]):
    raise ValueError("Missing required API keys in environment variables")

# Initialize components
@lru_cache(maxsize=None)
def initialize_components():
    """Initialize reusable components with caching"""
    log.info("Initializing Pinecone connection")
    pc = pinecone.Pinecone(api_key=app.config["PINECONE_API_KEY"])
    
    log.info("Downloading embeddings model")
    embeddings = HuggingFaceEmbeddings(model_name=app.config["EMBEDDING_MODEL"])
    
    log.info("Initializing vector store")
    vector_store = PineconeVectorStore(
        index=pc.Index(app.config["INDEX_NAME"]),
        embedding=embeddings
    )
    
    log.info("Creating prompt template")
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    log.info("Initializing LLM")
    # llm = ChatGroq(
    #     temperature=app.config["LLM_TEMPERATURE"],
    #     groq_api_key=app.config["GROQ_API_KEY"],
    #     model_name=app.config["LLM_MODEL"]
    # )

    
    llm=CTransformers(model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=app.config["SEARCH_CONFIG"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Initialize QA chain at app startup
qa_chain = initialize_components()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def handle_query():
    try:
        msg = request.form["msg"]
        input = msg
        print(f"User Query: {input}")
        response=qa_chain.invoke({"query": input})
        print("Response: ", response["result"])
        return str(response["result"])
        
    except Exception as e:
        log.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=os.getenv("FLASK_DEBUG", False))