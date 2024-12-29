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
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link & keep the model in the model directory:

Download the Llama 2 Model: llama-2-7b-chat.ggmlv3.q4_0.bin from the following link:

```ini
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone

