# MedicalChatbot-Llama2

## Installation Steps

### Step 1. Clone the repository

This command will clone the repository from GitHub to your local machine. 
```bash
git clone https://github.com/Gaurav0963/MedicalChatbot-Llama2.git
```
Change your directory to the cloned project directory.
```bash
cd MedicalChatbot-Llama2/
```

### Step 2. Create a Conda Environment:

This command creates a new Conda environment named `chatbot` with Python 3.9 installed.

```bash
conda create -n chatbot python=3.9 -y
```

### Step 3. Activate Conda Environment:
This command activates the `chatbot` Conda environment.
```bash
conda activate chatbot
```

### Step 4. Install The Requirements:
This command installs all the necessary Python packages listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### Step 5. Create `.env` File:
Create a `.env` file in the root directory and add your `Pinecone API key`. This file will store environment variables (your Pinecone credentials).

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 6. Download the Quantized Model:

Download the Llama 2 model file from the provided link and place it in the `model` directory.

```ini
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### Step 7. Create Pinecone Index:
Run this command to create the Pinecone index and upsert the embedded vectors to it. Skip this step if it has already been done previously.
```bash
python store_index.py
```

### Step 8. Run the Flask App:
This command starts the Flask application.
```bash
python app.py
```

Open your browser and navigate to `localhost:8080` to access the Flask app.
```bash
localhost:8080
```

## Tech Stack Used:

- `Python`: The main programming language for the project.

- `LangChain`: Used for language model operations.

- `Flask`: A web framework for creating the web application.

- `Meta Llama2`: The language model used for the chatbot.

- `Pinecone`: A vector database for similarity search.