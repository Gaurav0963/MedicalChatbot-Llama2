import getpass
import os
import time
from uuid import uuid4
import pinecone
from langchain_pinecone import PineconeVectorStore
from src.utils import PDFProcessor
from src.logger import logging as log
from src.exception import CustomException


def get_pinecone_api_key():
    try:
        if not os.getenv("PINECONE_API_KEY"):
            os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
        return os.environ.get("PINECONE_API_KEY")
    except Exception as e:
        raise CustomException(f"Error retrieving Pinecone API key: {str(e)}")


def initialize_pinecone(api_key):
    try:
        return pinecone.Pinecone(api_key=api_key)
    except Exception as e:
        raise CustomException(f"Error initializing Pinecone: {str(e)}")


def create_or_connect_index(pinecone_client, index_name, dimension):
    try:
        if index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pinecone_client.describe_index(index_name).status["ready"]:
                time.sleep(1)
        return pinecone_client.Index(index_name)
    except Exception as e:
        raise CustomException(f"Error creating or connecting to Pinecone index: {str(e)}")


def main():
    try:
        log.info("Starting the process...")

        # Get Pinecone API key
        pinecone_api_key = get_pinecone_api_key()
        log.info("Retrieved Pinecone API key.")

        # Initialise Pinecone
        pinecone_client = initialize_pinecone(api_key=pinecone_api_key)
        log.info("Initialized Pinecone client.")

        # Connect to Pinecone index
        index_name = "medical-chatbot"
        index = create_or_connect_index(pinecone_client, index_name, dimension=384)
        log.info(f"Connected to Pinecone index: {index_name}")

        # Initialize PDFProcessor
        dataset_directory = "data/"
        processor = PDFProcessor(data_directory=dataset_directory)
        log.info("Initialized PDFProcessor.")

        # Load text data from .pdf files
        processor.load_pdf()
        log.info("Loaded text data from PDF files.")

        # Create text chunks of the loaded text
        processor.text_split()
        log.info("Created text chunks from loaded text.")

        # Download embeddings model from Huggingface
        processor.download_embedding_model()
        log.info("Downloaded embeddings model from Huggingface.")

        # Initialize vector store
        vector_store = PineconeVectorStore(index=index, embedding=processor.embeddings)
        log.info("Initialized Pinecone vector store.")

        # Add vectors to vector store
        uuids = [str(uuid4()) for _ in range(len(processor.text_chunks))]
        vector_store.add_documents(documents=processor.text_chunks, ids=uuids)
        log.info("Added vectors to Pinecone vector store.")

        log.info("Process completed successfully.")
    except CustomException as ce:
        log.error(str(ce))
    except Exception as e:
        log.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
