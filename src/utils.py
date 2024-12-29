import sys
from src.logger import logging
from src.exception import CustomException
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Loading .pdf file from direcotry: data/Medical_Book.pdf
def load_pdf(data):
    try:
        logging.info("Loading pdf file...")
        loader = DirectoryLoader(
            data,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        document = loader.load()
        logging.info("Done.")
        return document
    except CustomException as e:
        logging.error("Couldn't load pdf file")
        raise CustomException(e, sys)


# Create text chunks with error handling and logging
def text_split(loaded_data):
    try:
        logging.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=20,
            length_function=len)
        text_chunks = text_splitter.split_documents(loaded_data)
        logging.info("Text splitting done.")
        return text_chunks
    except CustomException as e:
        logging.error(f"An error occurred during text splitting: {e}")
        raise CustomException(e, sys)


# Dowloading the embedding model from huggingface
def download_embedding_model():
    try:
        logging.info("Downloading embeddings model (all-MiniLM-L6-v2) from huggingface.")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings
    except CustomException as e:
        logging.error("Couldn't download embedding model")
        raise CustomException(e, sys)
 