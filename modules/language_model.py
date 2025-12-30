## import dependencies
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

## environment setup
import os
from dotenv import load_dotenv
load_dotenv()

## for gemini models
# google_key = os.getenv("GOOGLE_API_KEY", None)
# assert google_key, "[ATTENTION] GOOGLE API KEY IS MISSING!"
# os.environ["GOOGLE_API_KEY"] = google_key

## for cohere models
cohere_api = os.getenv("COHERE_API_KEY", None)
assert cohere_api, "[ATTENTION] COHERE API KEY IS MISSING!"
os.environ["COHERE_API_KEY"] = cohere_api

## chat model
llm = ChatCohere(temperature=0.0)

## embedding model
embedF = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## chat model for ragas evaluation
# ragas_eval_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
ragas_eval_llm = ChatOllama(model="qwen2.5:latest")
ragas_eval_embedF = OllamaEmbeddings(model="nomic-embed-text:latest")