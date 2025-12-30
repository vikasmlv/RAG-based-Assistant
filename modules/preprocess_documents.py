## import dependencies
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def load_chunk_store(data_path: str) -> List[Document]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
        )

    if not len(os.listdir(data_path)) > 0:
        raise FileNotFoundError(
            f"[ALERT] No file exists in {data_path}"
        )
    
    ## list of all the PDFs
    pdfs = [pdf for pdf in os.listdir(data_path) if pdf.endswith(".pdf")] ## list of all file names as str that ends with `.pdf`

    doc_container = [] ## list of all chunked documents

    ## take each item from `pdfs` and load it using PyPDFLoader
    for pdf in pdfs:
        loader = PyPDFLoader(file_path=os.path.join(data_path, pdf),
                                extract_images=False)
        docs_raw = loader.load() ## list of `Document` objects loaded page wise per PDF. Each such object has - 1. Page Content // 2. Metadata
        for doc in docs_raw:
            doc_container.append(doc) ## append each `Document` object to the previously declared container

    ## split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    prepped_docs = splitter.split_documents(documents=doc_container)
    return prepped_docs

## ---- TEST ----
if __name__=="__main__":
    folder_location = "./data/raw"
    prepped_docs = load_chunk_store(folder_location)
    print(prepped_docs, end="\n\n\n")
    print(f"Number of Document Chunks: {len(prepped_docs)}")
    # print(f"Number of Document Chunks: {len(docs)}")