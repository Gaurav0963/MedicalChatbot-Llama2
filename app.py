from flask import Flask, render_template, jsonify, request
import getpass
from src.utils import download_embedding_model
from src.logger import logging as log
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

from src.prompt import *
import os

app = Flask(__name__)

log.info("Getting credentials")
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialise the Pinecone
log.info("Initialise the Pinecone")
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Connect to pinecone index
log.info("Connecting to Pinecone Index...")
index_name = "medical-chatbot"
index = pc.Index(index_name)
log.info(f"Connected to Ponecone Index: {index_name}.")

# Download embeddings model from huggingface
log.info("Downloading embeddings model")
embeddings = download_embedding_model()
log.info("Embeddings model downloaded")

# Initialising vector store
log.info("Initialising vector store")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

log.info("Creating prompt")
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


log.info("creating llm model")
llm=CTransformers(model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# Query by turning into retriever
# Transforming the vector store into a retriever for easier usage in chains.
log.info("Query by turning into retriever")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)
log.info("RetrivalQA")
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
    )

@app.route("/")
def index():
    log.info("rendering chat.html")
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Query: {input}")
    response=qa.invoke({"query": input})
    print("Response: ", response["result"])
    return str(response["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

