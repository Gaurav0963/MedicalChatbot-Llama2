import sys
from src.logger import logging
from src.exception import CustomException
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


class PDFProcessor:
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.loaded_data = None
        self.text_chunks = None
        self.embeddings = None
    
    def load_pdf(self)-> None:
        '''
        This function loads text from PDF file'''
        try:
            logging.info("Loading pdf file...")
            print("Loading data...")
            loader = DirectoryLoader(
                self.data_directory,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            self.loaded_data = loader.load()
            logging.info("PDF file loaded successfully.")
            print("PDF file loaded successfully.")
        except CustomException as e:
            logging.error("Couldn't load pdf file")
            raise CustomException(e, sys)
    
    def text_split(self)-> None:
        '''
        This function slpits the loaded text from PDF file into text chunks with an overlap to maintain conetxt.
        '''
        try:
            logging.info("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            self.text_chunks = text_splitter.split_documents(self.loaded_data)
            logging.info("Text splitting done.")
        except CustomException as e:
            logging.error(f"An error occurred during text splitting: {e}")
            raise CustomException(e, sys)
    
    def download_embedding_model(self)-> None:
        '''
        This function downloads embeddings model (all-MiniLM-L6-v2) from huggingface.
        '''
        try:
            if not self.embeddings:
                logging.info("Downloading embeddings model (all-MiniLM-L6-v2) from huggingface.")
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except CustomException as e:
            logging.error("Couldn't download embedding model")
            raise CustomException(e, sys)
 