from typing import List
from langchain.schema import Document

class RRF:
    def __init__(self, documents: List[List[Document]]) -> None:
        self.documents = documents
        self.rrf_scores = dict()

    def rearrange(self, top_k: int = 0) -> List[Document]:
        doc_map = dict()
        for docs in self.documents:
            for rank, doc in enumerate(docs, start=1):
                ## making langchain.Schema.Document hashable
                key = (doc.page_content, tuple(doc.metadata.items()))
                doc_map[key] = doc

                rrf_rank = 1/(60+rank)
                self.rrf_scores[key] = self.rrf_scores.get(key, 0) + rrf_rank
        
        self.rrf_scores = sorted(self.rrf_scores.items(), key=(lambda x: x[1]), reverse=True)
        best_docs = [doc_map[key] for key, _ in self.rrf_scores]

        return best_docs[:top_k] if top_k else best_docs
    
## just for testing the module
if __name__=="__main__":
    query_one = [
        "this is document five", ## let's say one particular document is only retrieved against a particular query
        "this is document four",
        "this is document three",
        "this is document two",
    ]
    query_one = [Document(page_content=text) for text in query_one]

    query_two = [
        "this is document one",
        "this is document three",
        "this is document two",
        "this is document four",
    ]
    query_two = [Document(page_content=text) for text in query_two]

    query_three = [
        "this is document four",
        "this is document two",
        "this is document one",
        "this is document three",
    ]
    query_three = [Document(page_content=text) for text in query_three]

    all_docs = [query_one, query_two, query_three]

    rrf = RRF(all_docs)
    print(rrf.rearrange(3))

    # """
    # (rag-based-legal-assistant) C:\Users\sodey\Downloads\_projects\RAG-based-Legal-Assistant>uv run -m modules.rrf_score
    # [Document(metadata={}, page_content='this is document four'), Document(metadata={}, page_content='this is document three'), Document(metadata={}, page_content='this is document two')]
    # """