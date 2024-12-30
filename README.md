# MedicalChatbot-Llama2

# Installation

### Step 1. Clone the repository

```bash
Project repo: https://github.com/Gaurav0963/MedicalChatbot-Llama2.git
```
Change direcotry to project direcotry
```bash
cd to-ptoject-directory
```

### Step 2. Create a conda environment after opening the repository

```bash
conda create -n chatbot python=3.9 -y
```

```bash
conda activate chatbot
```

### Step 3. Install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link & keep the model in the model directory:

Download the Llama 2 Model: llama-2-7b-chat.ggmlv3.q4_0.bin from the following link:

```ini
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

Create Pinecone Index and upsert data (embedded vectors) to it, skip if already done.
```bash
python store_index.py
```

Command to run Flask app:
```bash
python app.py
```

Now Flask app by running following command in terminal:
```bash
localhost:8080
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone

