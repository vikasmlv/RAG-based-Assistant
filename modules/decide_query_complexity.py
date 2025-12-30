## import dependencies
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from modules.conversation_history import ConversationSummaryMemory

## prompt template
with open("./prompts/decide_query_complexity.md", "r", encoding="utf-8") as file:
    complexity_prompt = file.read()

class QueryComplexitySchema(BaseModel):
    complexity: Literal["simple_conversation", "complex", "multi_hop"] = Field(
        default="simple_conversation",
        description="""
        Indicates how the system should reason about and retrieve context for the user's query.

        `simple_conversation`:
        Purely conversational inputs—greetings, acknowledgments, chit-chat, or general dialogue.
        No retrieval or document processing is required.

        `complex`:
        Any user query that must be answered factually using the provided context or knowledge base.
        These queries trigger a MultiQuery retrieval workflow, where multiple reformulated queries
        are generated, documents are retrieved for each, and Reciprocal Rank Fusion (RRF) is applied
        to re-rank and optimize the final document set before answering.

        `multi_hop`:
        Queries requiring layered or chained reasoning across multiple distinct facts or documents.
        Retrieval occurs iteratively—each hop uses previously retrieved information to guide deeper,
        context-dependent discovery and inference.
        """
    )

## determine query complexity based on the user query and previous conversation
class QueryComplexity:
    def __init__(
            self,
            model,
            prompt_template: str = complexity_prompt,
            pydantic_schema = QueryComplexitySchema
    ):
        self.model = model
        self.template = prompt_template
        self.output_schema = pydantic_schema

    @staticmethod
    def __initiate_chain(template, model, pydantic_schema):
        parser = JsonOutputParser(pydantic_object=pydantic_schema)
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_query", "memory"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | model | parser
        return chain

    def invoke(self, user_query: str, memory: ConversationSummaryMemory):
        chain = self.__initiate_chain(
            template=self.template,
            model=self.model,
            pydantic_schema=self.output_schema
        )
        resp = chain.invoke(
            {
                "user_query": user_query,
                "memory": memory,
            }
        )
        return resp["complexity"]

## --- TEST EXECUTION ---
if __name__ == "__main__":
    from langchain_cohere import ChatCohere
    from langchain_core.messages import AIMessage, HumanMessage

    ## set up environment
    import os
    from dotenv import load_dotenv

    load_dotenv()

    cohere_api = os.getenv("COHERE_API_KEY")
    assert cohere_api, "[ALERT] COHERE API KEY IS NOT IN THE ENVIRONMENT!"
    os.environ['COHERE_API_KEY'] = cohere_api

    ## llm
    llm = ChatCohere(temperature=0.2)

    ## dummy messages
    messages = [
        AIMessage(content="Hi! How may I assist you today?"),
        HumanMessage(content="Who play Iron Man in the MCU?"),
        AIMessage(content="Robert Downey Jr. plays the Iron Man character in MCU."),
        HumanMessage(content="What are his some of the best works?"),
        AIMessage(content="His best works include Chaplin, Sherlock Holmes, The Judge, Oppenheimer and many more."),
        HumanMessage(content="Has he ever acted in a movie along with Tom Cruise?"),
        AIMessage(content="Yes, he worked in Tropic Thunder with Tom Cruise.")
    ]

    ## add the dummy messages to memory
    memory = ConversationSummaryMemory(model=llm, k=2)
    for turn in messages:
        memory.append(turn)

    latest_user_query = """
    Considering Robert Downey Jr.’s role in Tropic Thunder alongside Tom Cruise, can you identify another actor from that same film who later collaborated with Marvel Studios, specify the MCU project they were involved in, and explain how the genre and role type of that MCU appearance differed from their role in Tropic Thunder?
    """

    ## instantiate complexity object
    complexity_decider = QueryComplexity(model=llm)

    ## invoke to get the complexity tier
    inferred_complexity = complexity_decider.invoke(
        user_query=latest_user_query,
        memory=memory
    )

    print(f"Inferred complexity: {inferred_complexity}")

    # """
    # (rag-based-legal-assistant) C:\Users\sodey\Downloads\_projects\RAG-based-Legal-Assistant>uv run -m modules.decide_query_complexity
    # Inferred complexity: complex
    # """