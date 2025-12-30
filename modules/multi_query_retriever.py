## import dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from modules.conversation_history import ConversationSummaryMemory
from typing import List
from langchain.schema import Document
from modules.rrf_score import RRF

## multi query prompt
with open("./prompts/multiQuery-prompt.md", "r") as f:
    multiquery_template = f.read()

## defined schema for the multiquery queries
class MultiQuerySchema(BaseModel):
        generatedQueries: List[str] = Field(
            default=[], 
            description="""
            List of semantically equivalent query variations for improved retrieval coverage.
            Contains 5 alternative phrasings.
            """
        )

## defined class
class MultiQueryRetriever:
    def __init__(
            self,
            model,
            bm25_retriever,
            semantic_retriever,
            multiquery_prompt_template: str = multiquery_template,
            top_k=3,
            output_schema = MultiQuerySchema
    ) -> None:
        self.model = model
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.multiquery_prompt_template = multiquery_prompt_template
        self.output_schema = output_schema
        self.top_k = top_k

        self.sub_queries: List[str] = []
        self.all_documents: List[List[Document]] = []

    def __create_multiquery_chain(self):
        parser = JsonOutputParser(pydantic_object=self.output_schema)
        prompt = PromptTemplate(
            template=self.multiquery_prompt_template ,
            input_variables=["chat_history", "user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.model | parser
        return chain

    def __translate_queries(self, user_query, memory) -> None:
        chain = self.__create_multiquery_chain()
        resp = chain.invoke({"chat_history": memory, "user_query": user_query})
        self.sub_queries.extend(resp.get("generatedQueries", [""]) if isinstance(resp, dict) else [""])

    def __retrieve(self) -> None:
        for query in self.sub_queries:
            # print(f"> Subquery: {query}") ## for debugging
            relevant_docs = self.bm25_retriever.invoke(query) + self.semantic_retriever.invoke(query) ## List[Document]
            self.all_documents.append(relevant_docs)

    def invoke(self, user_query: str, memory: ConversationSummaryMemory) -> List[Document]:
        self.__translate_queries(user_query, memory)
        self.__retrieve()
        rrf = RRF(self.all_documents)
        best_docs = rrf.rearrange(top_k=self.top_k)
        return best_docs

##--------------------- test the module -------------------------------
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_cohere import ChatCohere
    from modules.preprocess_documents import load_chunk_store
    from modules.bm25_retriever import instantiate_bm25retriever
    from modules.semantic_retriever import SemanticRetriever
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    ## setup PATH
    INPUT_DATA_PATH = "./data/raw"
    OUTPUT_DATA_PATH = "./data/vectors"

    ## set up environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    cohere_key = os.getenv("COHERE_API_KEY", None)
    assert cohere_key, "[ATTENTION] Cohere API key is required."
    os.environ['COHERE_API_KEY'] = cohere_key

    dummy_chat_history = [
        HumanMessage(content="I'm studying the Special Marriage Act for a legal automation project."),
        AIMessage(
            content="Nice! The Act has many procedural layers — notice, objections, inquiry, solemnization, registration, and even penalties. What aspects are you focusing on?"),
        HumanMessage(
            content="Mainly the notice and objection procedures, especially how Section 5 notice publication works and what Marriage Officers must do if an objection is raised."),
        AIMessage(
            content="Good area. Section 6 requires the Marriage Officer to keep the notice open for public inspection, and Section 7 allows any person to object on grounds that the marriage violates Section 4 conditions."),
        HumanMessage(
            content="Yes, but I’m also trying to understand how the timelines interact — especially the 30-day waiting period and how the inquiry under Section 8 should be conducted."),
        AIMessage(
            content="Right. Section 8 gives the Marriage Officer civil-court powers for inquiry, and Section 9 bars solemnization until the objection is resolved. The 30-day window under Section 7 acts as the minimum period before solemnization."),
        HumanMessage(
            content="Great. I also want to test how my system handles multi-step reasoning across Sections 4, 5, 6, 7, and 8."),
        AIMessage(content="Sure. Tell me the exact kind of query you want me to generate.")
    ]

    latest_user_query = (
        "If someone objects to a Section 5 marriage notice claiming the groom already has a "
        "living spouse, what sections must the Marriage Officer consult to decide the objection, "
        "and can the marriage be solemnized before the inquiry is finished?"
    )
    llm = ChatCohere(temperature=0.2)

    ## create chat history
    memory = ConversationSummaryMemory(model=llm, k=2)
    for item in dummy_chat_history:
        memory.append(item)

    ## retrievers
    preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)
    sparse_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

    embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    dense_retriever = SemanticRetriever(
        embedding_function=embedF,
        prepped_docs=preprocessed_docs,
        vectordb_output_path=OUTPUT_DATA_PATH
    ).retriever

    ## multiquery
    mq = MultiQueryRetriever(model=llm, top_k=5, bm25_retriever=sparse_retriever, semantic_retriever=dense_retriever)
    print(f"> Best Docs:\n\n{mq.invoke(user_query=latest_user_query, memory= memory)}\n\n\n")
    print(f"> Subqueries: \n\n{mq.sub_queries}")

    # """
    # (rag-based-legal-assistant) C:\Users\sodey\Downloads\_projects\RAG-based-Legal-Assistant>uv run -m modules.multi_query_retriever
    # > Best Docs:
    #
    # [Document(metadata={'producer': 'Microsoft® Word 2010', 'creator': 'Microsoft® Word 2010', 'creationdate': '2019-11-15T11:19:05+04:00', 'author': 'Cash', 'moddate': '2019-11-15T11:19:05+04:00', 'source': './data\\special_marriage_act.pdf', 'total_pages': 25, 'page': 5, 'page_label': '6'}, page_content='marriage, the Marriage Officer shall not solemnize the marriage until he has inquired into the matter of \nthe objection and is satisfied that it ought not to prevent the solemnization of the marriage or the objection \nis withdrawn by the person making it; but the Marriage Officer shall not take more than thirty days from \nthe date of the objection for the purpose of inquiring into the matter of the objection and arriving at a \ndecision. \n(2) If the Marriage Officer upholds the objection and refuses to solemnize the marriage, either party \nto the intended marriage may, within a period of thirty days from the date of such refusal, prefer an appeal \nto the district court within the local li mits of whose jurisdiction the Marriage Officer has his office, and \nthe decision of the district court on such appeal shall be final, and the Marriage Officer shall act in \nconformity with the decision of the court.'), Document(id='791e0ac0-2251-4959-84dc-2a625b519b33', metadata={'producer': 'Microsoft® Word 2010', 'creator': 'Microsoft® Word 2010', 'creationdate': '2019-11-15T11:19:05+04:00', 'author': 'Cash', 'moddate': '2019-11-15T11:19:05+04:00', 'source': './data\\special_marriage_act.pdf', 'total_pages': 25, 'page': 0, 'page_label': '1'}, page_content='1 \n \nTHE SPECIAL MARRIAGE ACT, 1954 \n______ \nARRANGEMENT OF SECTIONS \n______ \nCHAPTER I \nPRELIMINARY \nSECTIONS \n1. Short title, extent and commencement.  \n2. Definitions. \n3. Marriage Officers. \nCHAPTER II \nSOLEMNIZATION OF SPECIAL MARRIAGES \n 4. Conditions relating to solemnization of special marriages. \n 5. Notice of intended marriage. \n 6. Marriage Notice Book and publication. \n 7. Objection to marriage. \n 8. Procedure on receipt of objection. \n 9. Powers of Marriage Officers in respect of inquiries. \n10. Procedure on receipt of objection by Marriage Officer abroad. \n11. Declaration by parties and witnesses. \n12. Place and form of solemnization. \n13. Certificate of marriage. \n14. New notice when marriage not solemnized within three months. \nCHAPTER III \nREGISTRATION OF MARRIAGES CELEBRATED IN OTHER FORMS \n15. Registration of marriages celebrated in other forms. \n16. Procedure for registration. \n17. Appeals from orders under section 16.'), Document(id='95699509-195c-43e6-927d-f90897d7293c', metadata={'producer': 'Microsoft® Word 2010', 'creator': 'Microsoft® Word 2010', 'creationdate': '2019-11-15T11:19:05+04:00', 'author': 'Cash', 'moddate': '2019-11-15T11:19:05+04:00', 'source': './data\\special_marriage_act.pdf', 'total_pages': 25, 'page': 6, 'page_label': '7'}, page_content='10. Procedure on receipt of objection by Marriage Officer abroad .―Where an objection is made \nunder section 7 to a Marriage Officer  1[in the State of Jammu and Kashmir in respect of an intended \nmarriage in the State], and the Marriage Officer, after making such inquiry into the matter as he thinks fit, \nentertains a doubt in respect thereof, he shall not solemnize the marriage but shall transmit the record with \nsuch statement respecting the matter as he thinks fit to the Central Government, and the Central \nGovernment, after making such inquiry into the matter and after ob taining such advice as it thinks fit, \nshall give its decision thereon in writing to the Marriage Officer who shall act in conformity with the \ndecision of the Central Government. \n11. Declaration by parties and witnesses.―Before the marriage is solemnized the parties and three \nwitnesses shall, in the presence of the Marriage Officer, sign a declaration in the form specified in the'), Document(metadata={'producer': 'Microsoft® Word 2010', 'creator': 'Microsoft® Word 2010', 'creationdate': '2019-11-15T11:19:05+04:00', 'author': 'Cash', 'moddate': '2019-11-15T11:19:05+04:00', 'source': './data\\special_marriage_act.pdf', 'total_pages': 25, 'page': 7, 'page_label': '8'}, page_content='condition shall be subject to any law, custom or usage having the force of law governing each of them \nwhich permits of a marriage between the two; and \n(f) the parties have been residing within the district of the Marriage Officer for a period of not less \nthan thirty days immediately preceding the date on which the application i s made to him for \nregistration of the marriage. \n16. Procedure for registration .―Upon receipt of an application signed by both the parties to the \nmarriage for the registration of their marriage under this Chapter the Marriage Officer shall give public \nnotice thereof in such manner as may be prescribed and after allowing a period of th irty days for \nobjections and after hearing any objection received within that period, shall, if satisfied that all the \nconditions mentioned in section 15 are fulfilled, enter a certificate of the marriage in the Marriage'), Document(id='3c864468-dddc-453f-ae04-0bca238b0583', metadata={'producer': 'Microsoft® Word 2010', 'creator': 'Microsoft® Word 2010', 'creationdate': '2019-11-15T11:19:05+04:00', 'author': 'Cash', 'moddate': '2019-11-15T11:19:05+04:00', 'source': './data\\special_marriage_act.pdf', 'total_pages': 25, 'page': 16, 'page_label': '17'}, page_content='17 \n \n44. Punishment of bigamy.―Every person whose marriage is solemnized under this Act and who, \nduring the lifetime of his or her wife or husband, contracts any other marriage shall be subject to the \npenalties provided in section 494 and section 495 of the Indian Penal Code (45  of 1860), for the offence \nof marrying again during the lifetime of a husband or wife, and the marriage so contracted shall be void. \n45. Penalty for signing false declaration or certificate .―Every person making, signing or attesting \nany declaration or cert ificate required by or under this Act containing a statement which is false and \nwhich he either knows or believes to be false or does not believe to be true shall be guilty of the offence \ndescribed in section 199 of the Indian Penal Code (45 of 1860). \n46. Penalty for wrongful action of Marriage Officer .―Any Marriage Officer who knowingly and \nwilfully solemnizes a marriage under this Act,―')]
    #
    #
    #
    # > Subqueries:
    #
    # ['Procedural steps for Marriage Officers handling bigamy objections under Sections 4-9 of the Marriage Act', 'Can a marriage be solemnized before resolving a Section 5 objection regarding bigamy?', 'Legal consequences of proceeding with marriage solemnization during an ongoing Section 8 inquiry', 'Timeline for resolving objections under Section 8 and its impact on the 30-day waiting period under Section 7', 'How do Sections 6 and 9 interact to prevent bigamous marriages during the objection resolution process?']
    # """