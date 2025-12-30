# RAG-based Legal Assistant Chatbot

A powerful, context-aware legal assistant chatbot built with **LangChain** and advanced **Retrieval-Augmented Generation (RAG)** techniques.

The application combines **dense and sparse retrieval** by leveraging traditional **BM25-based keyword search** alongside **semantic vector search**, ensuring both lexical precision and deep contextual relevance. It intelligently analyzes the complexity of each user query and dynamically selects the most appropriate retrieval strategy.

For complex reasoning tasks, the system employs **Multi-Hop retrieval** across multiple documents, while **Multi-Query retrieval** is used as a semantic query expansion technique to improve recall and coverage. Results from these retrieval strategies are consolidated using **Reciprocal Rank Fusion (RRF)**, ensuring that the most contextually relevant documents are prioritized and passed to the language model as grounding context for response generation.

Additionally, the application includes a **custom-built chat history management module**, developed as a robust replacement for LangChain’s deprecated `ConversationSummaryBufferMemory`. This module reliably maintains conversational context across interactions without relying on unstable or deprecated abstractions.


## Features

- **PDF Document Processing**: Automatically processes and indexes legal PDF documents
- **Sparse (BM25) Retriever**: keyword-based retrieval method that ranks documents using term frequency and inverse document frequency to capture exact lexical matches between the query and documents
- **Dense Retriever**: FAISS-backed vector retriever with the `all-MiniLM-L6-v2` sentence transformer embedding model to perform semantic similarity search, retrieving documents based on contextual meaning rather than exact keyword matches
- **Multi-Query Retrieval**: Semantic query expansion strategy that generates multiple paraphrased or reformulated queries from the original user input to improve retrieval recall and coverage.
- **Multi-Hop Retrieval**: Decomposes a complex query into intermediate steps, retrieving and chaining context from multiple documents to enable reasoning across dispersed information sources
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple retrieval methods by prioritizing documents that consistently rank highly across different retrievers
- **Intelligent Query Classification**: Determines whether document retrieval is needed based on the complexity inferred
- **Conversation History Awareness**: Maintains conversation context across multiple turns
- **Vector Database Storage**: Efficiently stores and retrieves document embeddings using FAISS
- **Command-Line Interface**: Interactive terminal-based chat interface
- **Responsible AI Disclaimers**: Clearly communicates that responses are not substitutes for legal advice

## Technical Stack

- **Framework**: LangChain
- **Interface**: Command-line terminal interface
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Chat Model**: Cohere (command-r model)
- **Vector Store**: FAISS
- **Document Processing**: LangChain's PyPDFLoader
- **Text Splitting**: LangChain's RecursiveCharacterTextSplitter
- **Evaluation**: RAGAS
- **Traceability**: LangSmith

## Prerequisites

```bash
python >= 3.12
pip or uv (recommended)
```

## Installation

### Option 1: Using UV (Recommended)

1. Install UV (if not already installed):
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/sougaaat/RAG-based-Legal-Assistant.git
cd RAG-based-Legal-Assistant
```

3. Install dependencies with UV:
```bash
uv sync
```

### Option 2: Using Traditional pip

1. Clone the repository:
```bash
git clone https://github.com/sougaaat/RAG-based-Legal-Assistant.git
cd RAG-based-Legal-Assistant
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

Set up your environment variables by editing the `.env` file:
```bash
# Edit .env with your API keys:
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, if using OpenAI models
```

**Note**: The `.env` file is included in the repository with empty values. Simply add your actual API keys to get started.

## Project Structure

```
RAG-based-Legal-Assistant/
├── .venv/                          # Virtual environment (local)
├── data/                           # Raw and processed legal documents for indexing
├── modules/                        # Core RAG and chatbot logic
│   ├── __init__.py
│   ├── _dependency_sequence.md     # Explainable dependency/execution flow notes
│   ├── bm25_retriever.py           # Sparse retriever based on BM25
│   ├── semantic_retriever.py       # Dense retriever (FAISS + sentence transformers)
│   ├── multi_query_retriever.py    # Multi-query (semantic query expansion) strategy
│   ├── multi_hop_retriever.py      # Multi-hop retrieval logic
│   ├── rrf_score.py                # Reciprocal Rank Fusion (RRF) scoring
│   ├── decide_query_complexity.py  # Decides retrieval strategy based on query complexity
│   ├── preprocess_documents.py    # Document loading & chunking
│   ├── conversation_history.py    # Conversation memory management
│   ├── chatbot_response.py         # Response orchestration and generation
│   ├── language_model.py           # LLM initialization and configuration
│   └── nltk_tokenizer_download.py  # NLTK tokenizer setup helper
├── outputs/                        # Generated outputs and demo responses
│   └── output-demo.md
├── prompts/                        # Prompt templates for the LLM
├── RAGAS-dataset/                  # Evaluation datasets and scoring artifacts               
│   ├── eval-dataset.csv            # Dataset to perform the evaluation on (generated using GPT)
│   ├── eval-dataset-final.csv      # Dataset containing answers and retrieved context for the evaluation dataset
│   └── eval-score.csv              # Dataset with evaluation scores
├── trash/                          # Temporary or discarded files
├── .env                            # Environment variables (ignored by git)
├── .env_example                    # Sample environment configuration
├── .gitignore
├── .python-version                 # Python version pin
├── app.py                          # Application entry point
├── pyproject.toml                  # Project metadata and dependency configuration
├── requirements.txt                # Dependency list (pip-compatible)
├── uv.lock                         # Locked dependencies (uv)
├── ragas_eval.py                   # RAGAS evaluation pipeline
├── ragas_score.py                  # RAGAS scoring utilities
├── test-cases.txt                  # Manual test cases for validation
└── README.md                       # Project documentation
```

## Usage

1. **Prepare your documents**: Place your legal PDF documents in the `/data/raw` directory

2. **Start the chatbot**: Run the main application:
```bash
uv run app.py
```

3. **Interact with the bot**: The terminal interface will start, and you can begin asking questions. Type `exit` to quit.

## How It Works

The application uses an advanced RAG pipeline with several sophisticated components:

### 1. **Document Processing**:
   - Loads PDF documents from the data directory using PyPDFLoader
   - Splits documents into manageable chunks using RecursiveCharacterTextSplitter
   - Creates embeddings using HuggingFace's `all-MiniLM-L6-v2` model
   - Stores embeddings in ChromaDB for efficient retrieval

### 2. **Query Processing & Retrieval**:
   - **Complexity Classification**: Uses a Pydantic model to determine the complexity of a user query.
   - **Multi-Query Generation**: For "complex" queries, generates multiple semantically equivalent variations and retrieves respective relevant documents.
   - **Multi-hop Retrieval**: Decomposes a "super_complex" user query into a sequence of intermediate sub-queries, iteratively retrieving and linking information from multiple documents so that context gathered in earlier steps guides subsequent retrieval for deeper, cross-document reasoning.
   - **Irrelevant Query Handling**: In case of an "Irrelevant" query, it simply returns a hard-coded instruction to guide the user to ask only legal questions.
   - **Reciprocal Ranking Fusion(RRF)**: Combines ranked results from multiple retrieval strategies by assigning higher importance to documents that appear consistently near the top across retrievers, producing a more robust and contextually relevant final document ranking.
   - **Conversational Awareness**: Incorporates chat history for contextual understanding.

### 3. **Response Generation**:
   - Uses Cohere's language model for generating responses
   - Provides clear, concise answers based on retrieved context
   - Maintains conversation history for follow-up questions
   - Includes appropriate legal disclaimers

## API Keys Required

- **Cohere API Key**: Required for the chat model (sign up at [Cohere](https://cohere.com/))
- **OpenAI API Key**: Optional, if you want to switch to OpenAI models

