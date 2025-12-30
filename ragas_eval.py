## import dependencies
import time
import pandas as pd
from modules.language_model import llm, embedF
from langchain_core.messages import AIMessage, HumanMessage
from modules.conversation_history import ConversationSummaryMemory
from modules.preprocess_documents import load_chunk_store
from modules.decide_query_complexity import QueryComplexity
from modules.chatbot_response import ChatbotResponse
from modules.bm25_retriever import instantiate_bm25retriever
from modules.semantic_retriever import SemanticRetriever
from modules.multi_query_retriever import MultiQueryRetriever
from modules.multi_hop_retriever import MultiHopRetriever
from typing import List
from langchain.schema import Document

## supress langchain warning
import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## setup PATH
INPUT_DATA_PATH = "./data"
OUTPUT_DATA_PATH = "./data-ingestion-local"
RAGAS_DATASET_PATH = "./RAGAS-dataset/eval-dataset.csv"

## main RAG prompt template
with open("./prompts/mainRAG-prompt.md") as f:
    rag_prompt_template = f.read()

## RAGAS evaluation dataset
eval_df = pd.read_csv(RAGAS_DATASET_PATH)

## to store responses for evaluation
answers: List[str] = []
retrieved_documents: List[List[str]] = []

## chatbot response generator
response_generator = ChatbotResponse(model=llm, rag_prompt_template=rag_prompt_template)

## query complexity decider
complexity_decider = QueryComplexity(model=llm)

## process documents
preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)

## set up retrievers - bm25, semantic
sparse_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

dense_retriever = SemanticRetriever(
    embedding_function=embedF,
    prepped_docs=preprocessed_docs,
    vectordb_output_path=OUTPUT_DATA_PATH
).retriever

count = 0
## ---- GENERATE ANSWER IN A LOOP FOR THE EVALUATION DATASET ----
for index, row in eval_df.iterrows():
    retrieved_docs: List[Document] = []
    subquery_plus_docs = ""

    ## conversation history
    chat_history = ConversationSummaryMemory(model=llm, k=3)
    chat_history.append(AIMessage(content="Hello! I'm a RAG-based legal assistant. How may I help you today?"))

    ## pick the question from the dataset
    ragas_question = row["question"]

    ## fetch the complexity tier
    complexity_tier = complexity_decider.invoke(
        user_query=ragas_question,
        memory=chat_history
    )

    ## set up both the retrievers multi-query and multi-hop
    multiquery_retriever = MultiQueryRetriever(
        model=llm,
        bm25_retriever=sparse_retriever,
        semantic_retriever=dense_retriever
    )

    multihop_retriever = MultiHopRetriever(
        model=llm,
        bm25_retriever=sparse_retriever,
        semantic_retriever=dense_retriever
    )

    ## fetch document based on the retriever type decided from `complexity_tier`
    if complexity_tier == "complex":
        retrieved_docs.extend(multiquery_retriever.invoke(user_query=ragas_question, memory=chat_history))
    elif complexity_tier == "multi-hop":
        subquery_plus_docs += multihop_retriever.invoke(user_query=ragas_question, memory=chat_history)

    ## append the question in the chat memory
    chat_history.append(HumanMessage(content=ragas_question))  ## store it in the history

    ## response generation
    ai_resp = response_generator.invoke(
        memory=chat_history,
        subquery_docs=subquery_plus_docs,
        documents=retrieved_docs
    )

    ## append response to answers
    answers.append(ai_resp)


    ## append retriever documents
    if complexity_tier == "complex":
        ## for retrieved documents (Multi Query)
        retrieved_docs_string_format: List[str] = [doc.page_content for doc in retrieved_docs]
        retrieved_documents.append(retrieved_docs_string_format)
    elif complexity_tier == "multi-hop":
        ## for subquery_plus_docs (Multi Hop)
        multihop_retrieved_docs: List[Document] = multihop_retriever.retrieved_respective_documents
        multihop_retrieved_docs_string_format: List[str] = [doc.page_content for doc in multihop_retrieved_docs]
        retrieved_documents.append(multihop_retrieved_docs_string_format)

    ## debug procedures
    count += 1
    print(f"Question {count}".center(80, "-"))
    print(f"> Human: {ragas_question}")
    print(f"> AI: {ai_resp}")

    time.sleep(60)

## save all this to a dataframe
eval_df['response'] = answers
eval_df['retrieved_contexts'] = retrieved_documents

eval_df.to_csv("./RAGAS-dataset/eval-dataset-final.csv", index=False)