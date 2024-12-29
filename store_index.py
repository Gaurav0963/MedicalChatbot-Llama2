from src.utils import text_split, load_pdf, download_embedding_model
from src.logger import logging as log
from src.exception import CustomException
import getpass
import os
import time
from uuid import uuid4
import pinecone
from langchain_pinecone import PineconeVectorStore


# Getting pinecone DB credentials
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialise the Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Connect to pinecone index, if index is not present then one will be created
index_name = "medical-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  #Needs modification according to your embedding model dimension, sentence-transformers/all-MiniLM-L6-v2 dimension is 384
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)

# Load text data from .pdf file 
extract_data = load_pdf("data/")

# Create text chunks of the loaded text
chunks = text_split(extract_data)

# Download embeddings model from huggingface
embeddings = download_embedding_model()

# Initialising vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Add vectors to vector store
uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)

