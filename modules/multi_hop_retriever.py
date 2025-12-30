## import dependencies
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import Dict, List

from ragas.messages import HumanMessage, AIMessage

from modules.conversation_history import ConversationSummaryMemory
from modules.rrf_score import RRF

with open("./prompts/multihop-prompt.md", "r", encoding="utf-8") as f:
    multihop_template = f.read()

class SubQuery(BaseModel):
    end_of_generation: bool = Field(
        default=False,
        description="""
        Indicates whether further query decomposition is required in a multi-hop
        retrieval process. Set this to True when the system has generated enough
        subqueries and gathered sufficient supporting documents, signaling that
        no additional subquery generation is needed.
        """
    )
    subquery: str = Field(
        default="",
        description="""
        The individual subquery derived from the user's original query. Each
        subquery should represent a logically independent step that contributes
        to answering the overall multi-hop question.
        """
    )

## define multi-hop retriever class
class MultiHopRetriever:
    def __init__(
            self,
            model,
            bm25_retriever,
            semantic_retriever,
            multi_hop_prompt_template: str = multihop_template,
            pydantic_schema = SubQuery
    ):
        self.multi_hop_prompt_template = multi_hop_prompt_template
        self.model = model
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.pydantic_schema = pydantic_schema

        ## queries and documents
        self.end_of_generation = False
        self.subqueries = []
        self.retrieved_respective_documents = []

    def generate_subquery_content(self) -> str:
        starting_substring = "### Subqueries + Retrieved Documents\n"
        content = ""
        for index, (subquery, list_of_documents) in enumerate({subquery: list_of_docs for subquery, list_of_docs in zip(self.subqueries, self.retrieved_respective_documents)}.items(), start=1):
            content += f"{index}. Subquery: {subquery}\n-> Relevant Documents:\n{'\n'.join(doc.page_content for doc in list_of_documents)}"
            content +="\n"
        return starting_substring+content

    @staticmethod
    def __initiate_chain(template, model, pydantic_schema) -> Dict:
        parser = JsonOutputParser(pydantic_object=pydantic_schema)
        prompt = PromptTemplate(
            template=template,
            input_variables=["max_iteration_allowed", "user_query", "memory", "subqueries_and_relevant_documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | model | parser
        return chain

    def invoke(self, user_query, memory: ConversationSummaryMemory, max_iteration_allowed=5):
        for i in range(max_iteration_allowed):
            # print(f"> Try: {i}") ## for debugging
            chain = self.__initiate_chain(
                template=self.multi_hop_prompt_template,
                model=self.model,
                pydantic_schema=self.pydantic_schema
            )
            resp = chain.invoke(
                {
                    "max_iteration_allowed": max_iteration_allowed,                    "memory": memory,
                    "user_query": user_query,
                    "memory": memory,
                    "subqueries_and_relevant_documents": self.generate_subquery_content() if len(self.subqueries) > 0 else ""
                }
            )

            ## for debugging
            print(f"> EoG Status: {resp['end_of_generation']}")
            print(f"> Subquery: {resp['subquery']}")

            if not resp['end_of_generation']:
                """
                If the LLM has to continue generating more
                subqueries:
                -> append every subquery
                -> retrieve relevant documents using both retriever
                -> pack it in a List[List[Document]]
                -> fetch best documents via RRF
                -> append it to `retrieved_respective_documents`
                """
                self.subqueries.append(resp['subquery'])
                rrf = RRF(
                    [
                        self.bm25_retriever.invoke(resp['subquery']),  ## output -> List[Document]
                        self.semantic_retriever.invoke(resp['subquery']),  ## output -> List[Document]
                    ]  ## becomes: List[List[Document]]
                )
                best_relevant_docs = rrf.rearrange(top_k=7) ## -> List[Document]
                self.retrieved_respective_documents.append(best_relevant_docs)
            else:
                break
        ## fetch subquery -> fetch relevant docs -> generate content -> return content
        return self.generate_subquery_content()

##---- TESTING EXECUTION ----
if __name__ == "__main__":
    from langchain_cohere import ChatCohere
    from modules.bm25_retriever import instantiate_bm25retriever
    from modules.semantic_retriever import SemanticRetriever
    from modules.preprocess_documents import load_chunk_store
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    INPUT_DATA_PATH = "./data/raw"
    OUTPUT_DATA_PATH = "./data/vectors"

    import os
    from dotenv import load_dotenv
    load_dotenv()
    cohere_key = os.getenv("COHERE_API_KEY", None)
    assert cohere_key, "[ATTENTION] COHERE API KEY IS MISSING!"
    os.environ["COHERE_API_KEY"] = cohere_key

    ## ---- TEST FOR THE ENTIRE MULTI-HOP RETRIEVAL SYSTEM ----
    llm = ChatCohere(temperature=0.2)
    preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)
    sparse_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

    embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    dense_retriever = SemanticRetriever(
        embedding_function=embedF,
        prepped_docs=preprocessed_docs,
        vectordb_output_path=OUTPUT_DATA_PATH
    ).retriever

    original_user_query = """
    Which section sets the age requirement, what are the exact ages required for male and female, and does C.D.’s age (17) permit any discretion or exception under the Act (including references to Schedules, provisos, or the Act’s explanations)?
    """

    ## conversation memory
    memory = ConversationSummaryMemory(model=llm, k=2)
    memory.append(AIMessage(content="Hi! How can I assist you today?"))

    ## multi-hop retriever
    multihop_retriever = MultiHopRetriever(
        model=llm,
        bm25_retriever=sparse_retriever,
        semantic_retriever=dense_retriever
    )

    multihop_response = multihop_retriever.invoke(
        user_query=original_user_query,
        memory=memory,
    )

    print("MultiHop Response".center(100,"-"))
    print(multihop_response)
    #
    # """
    # (rag-based-legal-assistant) C:\Users\sodey\Downloads\_projects\RAG-based-Legal-Assistant>uv run -m modules.multi_hop_retriever
    # > Try: 0
    # > EoG Status: False
    # > Subquery: Identify the section in the Act that specifies age requirements for males and females, including any exceptions or discretion for individuals aged 17.
    # > Try: 1
    # > EoG Status: False
    # > Subquery: Identify the specific section(s) in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17.
    # > Try: 2
    # > EoG Status: False
    # > Subquery: Identify the specific provisions in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17, particularly focusing on sections related to consent, marriage, and criminal liability.
    # > Try: 3
    # > EoG Status: False
    # > Subquery: Identify the specific provisions in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17, particularly focusing on sections related to consent, marriage, and criminal liability, with explicit reference to sections 64 and 99.
    # > Try: 4
    # > EoG Status: True
    # > Subquery:
    # -----------------------------------------MultiHop Response------------------------------------------
    # ### Subqueries + Retrieved Documents
    # 1. Subquery: Identify the section in the Act that specifies age requirements for males and females, including any exceptions or discretion for individuals aged 17.
    # -> Relevant Documents:
    # 77
    # 2023; that it did not fall within any of the general exceptions of the said Sanhita; and that it
    # did not fall within any of the five exceptions to section 99, or that, if it did fall within
    # Exception 1, one or other of the three provisos to that exception applied to it.
    # (b) A is charged under section 116 of the Bharatiya Nyaya Sanhita, 2023, with
    # voluntarily causing grievous hurt to B by means of an instrument for shooting. This is
    # equivalent to a statement that the case was not provided for by section 120 of the said
    # Sanhita, and that the general exceptions did not apply to it.
    # (c) A is accused of murder, cheating, theft, extortion, adultery or criminal intimidation,
    # or using a false property-mark. The charge may state that A committed murder, or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # 28.A consent is not such a consent as is intended by any section of this Sanhita,––
    # (a) if the consent is given by a person under fear of injury, or under a misconception
    # of fact, and if the person doing the act knows, or has reason to believe, that the
    # consent was given in consequence of such fear or misconception; or
    # (b) if the consent is given by a person who, from unsoundness of mind, or
    # intoxication, is unable to understand the nature and consequence of that to which he
    # gives his consent; or
    # (c) unless the contrary appears from the context, if the consent is given by a
    # person who is under twelve years of age.
    # 29.The exceptions in sections 25, 26 and 27 do not extend to acts which are offences
    # independently of any harm which they may cause, or be intended to cause, or be known to
    # be likely to cause, to the person giving the consent, or on whose behalf the consent is given.
    # Illustration.
    # Causing miscarriage (unless caused in good faith for the purpose of saving the life of the
    # 14
    # which he is, under any law relating to extradition, or otherwise, liable to be apprehended
    # or detained in custody in India; or
    # (i) who, being a released convict, commits a breach of any rule made under
    # sub-section (5) of section 394; or
    # (j) for whose arrest any requisition, whether written or oral, has been received
    # from another police officer, provided that the requisition specifies the person to be
    # arrested and the offence or other cause for which the arrest is to be made and it
    # appears therefrom that the person might lawfully be arrested without a warrant by the
    # officer who issued the requisition.
    # (2) Subject to the provisions of section 39, no person concerned in a non-cognizable
    # offence or against whom a complaint has been made or credible information has been
    # received or reasonable suspicion exists of his having so concerned, shall be arrested except
    # under a warrant or order of a Magistrate.
    # such person as the Magistrate may from time to time direct:
    # Provided that the Judicial Magistrate may order the father of a minor female child
    # referred to in clause (b) to make such allowance, until she attains her majority, if the Judicial
    # Magistrate is satisfied that the husband of such minor female child, if married, is not
    # possessed of sufficient means:
    # Security for
    # unexpired
    # period of
    # bond.
    # Order for
    # maintenance
    # of wives,
    # children and
    # parents.
    # 5
    # 10
    # 15
    # 20
    # 25
    # 30
    # 35
    # 40
    # 45
    # 98.Whoever sells, lets to hire, or otherwise disposes of any child with intent that such
    # child shall at any age be employed or used for the purpose of prostitution or illicit intercourse
    # with any person or for any unlawful and immoral purpose, or knowing it to be likely that such
    # child will at any age be employed or used for any such purpose, shall be punished with
    # imprisonment of either description for a term which may extend to ten years, and shall also be
    # liable to fine.
    # Explanation 1.—When a female under the age of eighteen years is sold, let for hire, or
    # otherwise disposed of to a prostitute or to any person who keeps or manages a brothel, the
    # person so disposing of such female shall, until the contrary is proved, be presumed to have
    # disposed of her with the intent that she shall be used for the purpose of prostitution.
    # Explanation 2.—For the purposes of this section “illicit intercourse” means sexual
    # 18
    #
    # (b) the manner in which a Marriage Officer may hold inquiries under this Act and the procedure
    # therefor;
    # (c) the form and manner in which any books required by or under this Act shall be maintained;
    # (d) the fees that may be levied for the performance of any duty imposed upon a Marriage Officer
    # under this Act;
    # (e) the manner in which public notice shall be given under section 16;
    # (f) the form in which, and the intervals within which, copies of entries in the Marriage Certificate
    # Book shall be sent in pursuance of section 48;
    # (g) any other matter which may be or requires to be prescribed.
    # 1[(3) Every rule made by the Central Government under this Act shall be laid, as soon as may be after
    # it is made, before each House of Parliament, while it is in session, for a total period of thirty days which
    # may be comprised in one session or in two or more successive sessions, and if, before the expiry of the
    # village;
    # (c) a copy thereof shall be affixed to some conspicuous part of the
    # Court-house;
    # (ii) the Court may also, if it thinks fit, direct a copy of the proclamation to be
    # published in a daily newspaper circulating in the place in which such person ordinarily
    # resides.
    # (3) A statement in writing by the Court issuing the proclamation to the effect that the
    # proclamation was duly published on a specified day, in the manner specified in clause (i) of
    # sub-section (2), shall be conclusive evidence that the requirements of this section have
    # been complied with, and that the proclamation was published on such day.
    # (4) Where a proclamation published under sub-section ( 1) is in respect of a person
    # accused of an offence which is made punishable with imprisonment of ten years or more, or
    # imprisonment for life or with death under the Bharatiya Nyaya Sanhita, 2023 or under any
    # other law for the time being in force, and such person fails to appear at the specified place
    # 2. Subquery: Identify the specific section(s) in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17.
    # -> Relevant Documents:
    # 4. (1) All offences under the Bharatiya Nyaya Sanhita, 2023 shall be investigated,
    # inquired into, tried, and otherwise dealt with according to the provisions hereinafter
    # contained.
    # (2) All offences under any other law shall be investigated, inquired into, tried, and
    # otherwise dealt with according to the same provisions, but subject to any enactment for the
    # time being in force regulating the manner or place of investigating, inquiring into, trying or
    # otherwise dealing with such offences.
    # 5. Nothing contained in this Sanhita shall, in the absence of a specific provision to
    # the contrary, affect any special or local law for the time being in force, or any special
    # jurisdiction or power conferred, or any special form of procedure prescribed, by any other
    # law for the time being in force.
    # CHAPTER II
    # C
    # ONSTITUTION OF CRIMINAL COURTS AND OFFICES
    # 6. Besides the High Courts and the Courts constituted under any law, other than this
    # 77
    # 2023; that it did not fall within any of the general exceptions of the said Sanhita; and that it
    # did not fall within any of the five exceptions to section 99, or that, if it did fall within
    # Exception 1, one or other of the three provisos to that exception applied to it.
    # (b) A is charged under section 116 of the Bharatiya Nyaya Sanhita, 2023, with
    # voluntarily causing grievous hurt to B by means of an instrument for shooting. This is
    # equivalent to a statement that the case was not provided for by section 120 of the said
    # Sanhita, and that the general exceptions did not apply to it.
    # (c) A is accused of murder, cheating, theft, extortion, adultery or criminal intimidation,
    # or using a false property-mark. The charge may state that A committed murder, or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # 53
    # and the substance thereof shall be entered in a book to be kept by such officer in such form
    # as the State Government may prescribe in this behalf:
    # Provided that if the information is given by the woman against whom an offence
    # under section 64, section 66, section 67, section 68, section 70, section 73, section 74,
    # section 75, section 76, section 77, section 78 or section 122 of the Bharatiya Nyaya
    # Sanhita, 2023 is alleged to have been committed or attempted, then such information shall
    # be recorded, by a woman police officer or any woman officer:
    # Provided further that—
    # (a) in the event that the person against whom an offence under section 354,
    # section 67, section 68, sub-section ( 2) of section 69, sub-section ( 1) of section 70,
    # section 71, section 74, section 75, section 76, section 77  or section 79 of the Bharatiya
    # Nyaya Sanhita, 2023 is alleged to have been committed or attempted, is temporarily or
    # with imprisonment for a term exceeding six months, it appears to the Magistrate that in the
    # interests of justice, the offence should be tried in accordance with the procedure for the trial
    # of warrant-cases, such Magistrate may proceed to re-hear the case in the manner provided
    # by this Sanhita for the trial of warrant-cases and may re-call any witness who may have
    # been examined.
    # CHAPTER XXIII
    # S
    # UMMARY TRIALS
    # 283. (1) Notwithstanding anything contained in this Sanhita—
    # (a) any Chief Judicial Magistrate;
    # (b) Magistrate of the first class,
    # shall try in a summary way all or any of the following offences:—
    # (i) theft, under section 301, section 303 or section 304 of the Bharatiya
    # Nyaya Sanhita, 2023 where the value of the property stolen does not exceed
    # twenty thousand rupees;
    # (ii) receiving or retaining stolen property, under section 315 of the
    # Bharatiya Nyaya Sanhita, 2023, where the value of the property does not exceed
    # twenty thousand rupees;
    # that sub-section shall, unless the contrary is proved, be presumed to be genuine and shall
    # be received in evidence.
    # (6) No Court shall take cognizance of an offence under section 64 of the Bharatiya
    # Nyaya Sanhita, 2023, where such offence consists of sexual intercourse by a man with his
    # own wife, the wife being under eighteen years of age, if more than one year has elapsed from
    # the date of the commission of the offence.
    # Prosecution
    # for offences
    # against
    # marriage.
    # 5
    # 10
    # 15
    # 20
    # 25
    # 30
    # 35
    # 40
    # 45
    # defamation under the Bhartiya Nyaya Sanhita, 2023,
    # (ii) makes, produces, publishes or keeps for sale, imports, exports, conveys,
    # sells, lets to hire, distributes, publicly exhibits or in any other manner puts into
    # circulation any obscene matter such as is referred to in section 292 of the Bhartiya
    # Nyaya Sanhita, 2023,
    # and the Magistrate is of opinion that there is sufficient ground for proceeding, the Magistrate
    # may, in the manner hereinafter provided, require such person to show cause why he should
    # not be ordered to execute a bond, with or without sureties, for his good behaviour for such
    # period, not exceeding one year, as the Magistrate thinks fit.
    # (2) No proceedings shall be taken under this section against the editor, proprietor,
    # printer or publisher of any publication registered under, and edited, printed and published
    # in conformity with, the rules laid down in the Press and Registration of Periodicals Act, 2023
    # (2) The offences punishable under the sections of the Bharatiya Nyaya Sanhita
    # specified in the first two columns of the Table next following may, with the permission of the
    # Court before which any prosecution for such offence is pending, be compounded by the
    # persons mentioned in the third column of that Table:—
    #                        1 2                       3
    # 5
    # 10
    # 15
    # 20
    # 25
    # 30
    # 35
    # 40
    # 3. Subquery: Identify the specific provisions in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17, particularly focusing on sections related to consent, marriage, and criminal liability.
    # -> Relevant Documents:
    # 4. (1) All offences under the Bharatiya Nyaya Sanhita, 2023 shall be investigated,
    # inquired into, tried, and otherwise dealt with according to the provisions hereinafter
    # contained.
    # (2) All offences under any other law shall be investigated, inquired into, tried, and
    # otherwise dealt with according to the same provisions, but subject to any enactment for the
    # time being in force regulating the manner or place of investigating, inquiring into, trying or
    # otherwise dealing with such offences.
    # 5. Nothing contained in this Sanhita shall, in the absence of a specific provision to
    # the contrary, affect any special or local law for the time being in force, or any special
    # jurisdiction or power conferred, or any special form of procedure prescribed, by any other
    # law for the time being in force.
    # CHAPTER II
    # C
    # ONSTITUTION OF CRIMINAL COURTS AND OFFICES
    # 6. Besides the High Courts and the Courts constituted under any law, other than this
    # 77
    # 2023; that it did not fall within any of the general exceptions of the said Sanhita; and that it
    # did not fall within any of the five exceptions to section 99, or that, if it did fall within
    # Exception 1, one or other of the three provisos to that exception applied to it.
    # (b) A is charged under section 116 of the Bharatiya Nyaya Sanhita, 2023, with
    # voluntarily causing grievous hurt to B by means of an instrument for shooting. This is
    # equivalent to a statement that the case was not provided for by section 120 of the said
    # Sanhita, and that the general exceptions did not apply to it.
    # (c) A is accused of murder, cheating, theft, extortion, adultery or criminal intimidation,
    # or using a false property-mark. The charge may state that A committed murder, or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # that sub-section shall, unless the contrary is proved, be presumed to be genuine and shall
    # be received in evidence.
    # (6) No Court shall take cognizance of an offence under section 64 of the Bharatiya
    # Nyaya Sanhita, 2023, where such offence consists of sexual intercourse by a man with his
    # own wife, the wife being under eighteen years of age, if more than one year has elapsed from
    # the date of the commission of the offence.
    # Prosecution
    # for offences
    # against
    # marriage.
    # 5
    # 10
    # 15
    # 20
    # 25
    # 30
    # 35
    # 40
    # 45
    # 2
    #
    # Novelty and Non obviousness. It provides protection for the invention
    # to the owner of the patent for a limited period, i.e 20 years.
    # 2. Trademarks - A trademark is a di stinctive sign which identifies certain
    # goods or services as those produced or provided by a specific person or
    # enterprise.1 It may be one or a combination of words, letters, and
    # numerals.2
    # 3. Copyright and related rights -  Copyright is a legal term describing
    # rights given to creators for their literary and artistic works. 3 Creators
    # often sell the rights to their works to individuals or companies best able
    # to market the works in return for payment. These payments are often
    # made dependent on the actual use of the work, and are then referred to
    # as royalties.
    # 4. Geographic indications of source - A Geographical Indication (GI) is a
    # sign used on goods that have a specific geographical origin and possess
    # qualities, reputation or characteristics that are essent ially attributable to
    # 53
    # and the substance thereof shall be entered in a book to be kept by such officer in such form
    # as the State Government may prescribe in this behalf:
    # Provided that if the information is given by the woman against whom an offence
    # under section 64, section 66, section 67, section 68, section 70, section 73, section 74,
    # section 75, section 76, section 77, section 78 or section 122 of the Bharatiya Nyaya
    # Sanhita, 2023 is alleged to have been committed or attempted, then such information shall
    # be recorded, by a woman police officer or any woman officer:
    # Provided further that—
    # (a) in the event that the person against whom an offence under section 354,
    # section 67, section 68, sub-section ( 2) of section 69, sub-section ( 1) of section 70,
    # section 71, section 74, section 75, section 76, section 77  or section 79 of the Bharatiya
    # Nyaya Sanhita, 2023 is alleged to have been committed or attempted, is temporarily or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # Nyaya Sanhita, 2023; but the sections under which the offence is punishable must, in each
    # instance be referred to in the charge.
    # (d) A is charged under section 220 of the Bharatiya Nyaya Sanhita, 2023, with
    # intentionally obstructing a sale of property offered for sale by the lawful authority of a
    # public servant. The charge should be in those words.
    # 235. (1) The charge shall contain such particulars as to the time and place of the
    # alleged offence, and the person (if any) against whom, or the thing (if any) in respect of
    # which, it was committed, as are reasonably sufficient to give the accused notice of the
    # matter with which he is charged.
    # (2) When the accused is charged with criminal breach of trust or dishonest
    # (iii) has been subject to recurrent attacks of insanity 4* * *;]
    # (c) the male has completed the age of twenty-one years and the female the age of eighteen years;
    # 5[(d) the parties are not within the degrees of prohibited relationship:
    # Provided that where a custom governing at least one of the parties permits of a marriage between
    # them, such marriage may be solemnized, notwithstanding that they are within the degrees of
    # prohibited relationship; and]
    # 6[(e) where the marriage is solemnized in the State of Jammu and Kashmir,  both parties are
    # citizens of India domiciled in the territories to which this Act extends].
    # 7[Explanation.―In this section, “custom”, in relation to a person belonging to any tribe, community,
    # group or family, means any rule which the State Government may, by notification in the Official Gazette,
    # specify in this behalf as applicable to members of that tribe, community, group or family:
    # 4. Subquery: Identify the specific provisions in the Bharatiya Nyaya Sanhita, 2023, that define age requirements for males and females, including any exceptions or discretion for individuals aged 17, particularly focusing on sections related to consent, marriage, and criminal liability, with explicit reference to sections 64 and 99.
    # -> Relevant Documents:
    # 4. (1) All offences under the Bharatiya Nyaya Sanhita, 2023 shall be investigated,
    # inquired into, tried, and otherwise dealt with according to the provisions hereinafter
    # contained.
    # (2) All offences under any other law shall be investigated, inquired into, tried, and
    # otherwise dealt with according to the same provisions, but subject to any enactment for the
    # time being in force regulating the manner or place of investigating, inquiring into, trying or
    # otherwise dealing with such offences.
    # 5. Nothing contained in this Sanhita shall, in the absence of a specific provision to
    # the contrary, affect any special or local law for the time being in force, or any special
    # jurisdiction or power conferred, or any special form of procedure prescribed, by any other
    # law for the time being in force.
    # CHAPTER II
    # C
    # ONSTITUTION OF CRIMINAL COURTS AND OFFICES
    # 6. Besides the High Courts and the Courts constituted under any law, other than this
    # 77
    # 2023; that it did not fall within any of the general exceptions of the said Sanhita; and that it
    # did not fall within any of the five exceptions to section 99, or that, if it did fall within
    # Exception 1, one or other of the three provisos to that exception applied to it.
    # (b) A is charged under section 116 of the Bharatiya Nyaya Sanhita, 2023, with
    # voluntarily causing grievous hurt to B by means of an instrument for shooting. This is
    # equivalent to a statement that the case was not provided for by section 120 of the said
    # Sanhita, and that the general exceptions did not apply to it.
    # (c) A is accused of murder, cheating, theft, extortion, adultery or criminal intimidation,
    # or using a false property-mark. The charge may state that A committed murder, or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # 53
    # and the substance thereof shall be entered in a book to be kept by such officer in such form
    # as the State Government may prescribe in this behalf:
    # Provided that if the information is given by the woman against whom an offence
    # under section 64, section 66, section 67, section 68, section 70, section 73, section 74,
    # section 75, section 76, section 77, section 78 or section 122 of the Bharatiya Nyaya
    # Sanhita, 2023 is alleged to have been committed or attempted, then such information shall
    # be recorded, by a woman police officer or any woman officer:
    # Provided further that—
    # (a) in the event that the person against whom an offence under section 354,
    # section 67, section 68, sub-section ( 2) of section 69, sub-section ( 1) of section 70,
    # section 71, section 74, section 75, section 76, section 77  or section 79 of the Bharatiya
    # Nyaya Sanhita, 2023 is alleged to have been committed or attempted, is temporarily or
    # cheating, or theft, or extortion, or adultery, or criminal intimidation, or that he used a false
    # property-mark, without reference to the definitions, of those crimes contained in the Bharatiya
    # Nyaya Sanhita, 2023; but the sections under which the offence is punishable must, in each
    # instance be referred to in the charge.
    # (d) A is charged under section 220 of the Bharatiya Nyaya Sanhita, 2023, with
    # intentionally obstructing a sale of property offered for sale by the lawful authority of a
    # public servant. The charge should be in those words.
    # 235. (1) The charge shall contain such particulars as to the time and place of the
    # alleged offence, and the person (if any) against whom, or the thing (if any) in respect of
    # which, it was committed, as are reasonably sufficient to give the accused notice of the
    # matter with which he is charged.
    # (2) When the accused is charged with criminal breach of trust or dishonest
    # that sub-section shall, unless the contrary is proved, be presumed to be genuine and shall
    # be received in evidence.
    # (6) No Court shall take cognizance of an offence under section 64 of the Bharatiya
    # Nyaya Sanhita, 2023, where such offence consists of sexual intercourse by a man with his
    # own wife, the wife being under eighteen years of age, if more than one year has elapsed from
    # the date of the commission of the offence.
    # Prosecution
    # for offences
    # against
    # marriage.
    # 5
    # 10
    # 15
    # 20
    # 25
    # 30
    # 35
    # 40
    # 45
    # charge. A may be separately charged with, and convicted of, two offences under section
    # 246 of the Bharatiya Nyaya Sanhita, 2023.
    # (f) A, with intent to cause injury to B, falsely accuses him of having committed an
    # offence, knowing that there is no just or lawful ground for such charge. On the trial, A gives
    # false evidence against B, intending thereby to cause B to be convicted of a capital offence.
    # A may be separately charged with, and convicted of, offences under sections 246 and 228
    # of the Bharatiya Nyaya Sanhita, 2023.
    # (g) A, with six others, commits the offences of rioting, grievous hurt and assaulting a
    # public servant endeavouring in the discharge of his duty as such to suppress the riot. A
    # may be separately charged with, and convicted of, offences under sections 189, 115 and 193
    # of the Bharatiya Nyaya Sanhita, 2023.
    # (h) A threatens B, C and D at the same time with injury to their persons with intent to
    # notification, specify in this behalf, upon any matter or thing duly submitted to him for
    # examination and report in the course of any proceeding under this Sanhita, may be used as
    # evidence in any inquiry, trial or other proceeding under this Sanhita, although such officer
    # is not called as a witness.
    # (2) The Court may, if it thinks fit, summon and examine any such officer as to the
    # subject-matter of his report:
    # Provided that no such officer shall be summoned to produce any records on which
    # the report is based.
    # (3) Without prejudice to the provisions of sections 129 and 130 of the Bharatiya
    # Sakshya Adhiniyam, 2023, no such officer shall, except with the permission of the General
    # Manager or any officer in charge of any Mint or of any Note Printing Press or of any
    # Security Printing Press or of any Forensic Department or any officer in charge of the Forensic
    # Science Laboratory or of the Government Examiner of Questioned Documents Organisation
    # """