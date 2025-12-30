## import langchain dependencies
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from modules.preprocess_documents import load_chunk_store

## BM25 retriever
def instantiate_bm25retriever(documents, tokenizer = None):
    if not tokenizer:
        return BM25Retriever.from_documents(documents=documents, k=10)
    else:
        return BM25Retriever.from_documents(documents=documents, preprocess_func=tokenizer, k=10)

# ---------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    ## load and chunk
    preprocessed_docs = load_chunk_store(data_path="./data/raw")

    ## BM25 retriever
    bm25_retriever = instantiate_bm25retriever(documents=preprocessed_docs, tokenizer=word_tokenize)

    ## user query
    user_query = input("> Human: ")

    ## result
    retrieved_docs = bm25_retriever.invoke(user_query)

    ## print result
    print("\n\n> Retriever Docs: ")
    for index, docs in enumerate(retrieved_docs, start=1):
        print(f"{index}:\n\n\n{docs.page_content}")
    print(f"Output:{type(retrieved_docs)}")
    print(f"Output:{type(retrieved_docs[0])}")