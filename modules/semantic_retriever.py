## langchain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.schema import Document

## settings up the env
import os
from dotenv import load_dotenv
load_dotenv()

class SemanticRetriever:
    """
    Semantic Retriever for FAISS:
    -> create an instance with embedding model, prepped documents and output path to save the vector store locally
    -> access the semantic retriever by .retriever
    -> invoke by .retriever.invoke(user_query)
    """
    def __init__(
            self,
            embedding_function: HuggingFaceEmbeddings,
            prepped_docs: List[Document],
            vectordb_output_path: str
    ) -> None:
        self.embedding_function = embedding_function
        self.prepped_docs = prepped_docs
        self.vectordb_output_path = os.path.abspath(vectordb_output_path)

    def __build_vectordb(self) -> None:
        vector_db = FAISS.from_documents(documents=self.prepped_docs, embedding=self.embedding_function)
        vector_db.save_local(self.vectordb_output_path)

    @property
    def retriever(self):
        if not os.path.exists(self.vectordb_output_path):
            self.__build_vectordb()
        ## if already exists, skip building
        vector_db = FAISS.load_local(self.vectordb_output_path, embeddings=self.embedding_function, allow_dangerous_deserialization=True)
        semantic_retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k": 10})
        return semantic_retriever
    
## ---- TEST ----
if __name__=="__main__":
    from modules.preprocess_documents import load_chunk_store
    import shutil

    INPUT_PATH = "./data/raw"
    DUMMY_OUTPUT_PATH = "./data/vectors/dummy"

    preprocessed_docs = load_chunk_store(data_path=INPUT_PATH)
    embedF = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    dense_retriever = SemanticRetriever(
        embedding_function=embedF,
        prepped_docs=preprocessed_docs,
        vectordb_output_path=DUMMY_OUTPUT_PATH
    ).retriever

    user_query = "Suppose a marriage is challenged as void under Section 24 on the ground that one party was already married, while the couple simultaneously seeks registration of an earlier ceremony under Chapter III claiming long-term cohabitation. How should the decision-maker reconcile the conditions in Section 15(b) (requiring no existing spouse) with the legitimacy protections for children under Section 26, exitand what procedural consequences follow if an appeal under Section 17 is already"

    relevant_documents = dense_retriever.invoke(user_query)

    for index, doc in enumerate(relevant_documents, start=1):
        print(f"{index}:: \n{doc.page_content}")

    ## once done delete the vector db
    shutil.rmtree(DUMMY_OUTPUT_PATH)