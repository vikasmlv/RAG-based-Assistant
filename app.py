## import dependencies
import time
import os
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

## supress langchain warning
import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## setup PATH
INPUT_DATA_PATH = "./data/raw"
OUTPUT_DATA_PATH = "./data/vectors"

## main RAG prompt template
with open("./prompts/mainRAG-prompt.md") as f:
    rag_prompt_template = f.read()

## conversation history
chat_history = ConversationSummaryMemory(model=llm, k=3)

## chatbot response generator
response_generator = ChatbotResponse(model=llm, rag_prompt_template=rag_prompt_template)

## query complexity decider
complexity_decider = QueryComplexity(model=llm)

## process documents
preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)

## set up retrievers - bm25, semantic and multi-hop(hybrid)
sparse_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

dense_retriever = SemanticRetriever(
    embedding_function=embedF,
    prepped_docs=preprocessed_docs,
    vectordb_output_path=OUTPUT_DATA_PATH
).retriever

## ---- CONVERSATION STARTER ----
welcome_message = "Welcome to the Legal Assistant Bot. How can I help you today? Write `exit` to quit."
chat_history.append(AIMessage(content=welcome_message))

## print welcome message
print("\033[1m> AI:\033[0m ", end="")
for letter in welcome_message:
    print(letter, flush=True, end="")
    time.sleep(0.03)

### ------------- CONVERSATION LOOP ----------------
while True:
    try:
        retrieved_docs = []
        subquery_plus_docs = ""

        ## user input
        print("\n\033[1m> Human:\033[0m ", end="")
        latest_user_query = input().strip()

        ## check if user wants to exit
        if latest_user_query.lower().strip() == "exit":
            bye_message = "Chat ended! See ya."
            print("\033[1m> AI:\033[0m ", end="")
            for letter in bye_message:
                print(letter, flush=True, end="")
                time.sleep(0.03)
            break

        ## fetch complexity level
        complexity_tier = complexity_decider.invoke(
            user_query=latest_user_query,
            memory=chat_history
        )

        ## instantiate retrievers for every turn
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
            retrieved_docs.extend(multiquery_retriever.invoke(user_query=latest_user_query, memory=chat_history))
        elif complexity_tier == "multi-hop":
            subquery_plus_docs += multihop_retriever.invoke(user_query=latest_user_query, memory=chat_history)
        ## forgot to handle "Irrelevant" complexity tier -> for now this is the way to go, will fix later
        else:
            ai_resp = "I'm here to assist you with Legal queries. Please refrain yourself from asking question outside this scope and chit-chatting."

            ## print AI response
            print(f"\033[1m> AI:\033[0m ({complexity_tier}) ", end="")
            for letter in ai_resp:
                print(letter, flush=True, end="")
                time.sleep(0.03)

            ## append AI response to chat history
            chat_history.append(AIMessage(content=ai_resp))

            continue

        ## STORE IN CHAT MEMORY
        chat_history.append(HumanMessage(content=latest_user_query))  ## store it in the history

        # print(f">>> {complexity_tier}") ## for debugging
        ## response generation
        ai_resp = response_generator.invoke(
            memory=chat_history,
            subquery_docs=subquery_plus_docs,
            documents=retrieved_docs
        )

        ## print AI response
        print(f"\033[1m> AI:\033[0m ({complexity_tier}) ", end="")
        for letter in ai_resp:
            print(letter, flush=True, end="")
            time.sleep(0.03)

        ## append AI response to chat history
        chat_history.append(AIMessage(content=ai_resp))

    except KeyboardInterrupt:
        print("\n\n[ALERT] ^C: Keyboard Interruption detected. Session Terminated!!")
        break