from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from modules.conversation_history import ConversationSummaryMemory
from typing import List

class ChatbotResponse:
    def __init__(self, model, rag_prompt_template) -> None:
        self.model = model
        self.rag_prompt_template = rag_prompt_template

    def invoke(
            self,
            memory: ConversationSummaryMemory,
            documents: List[Document] = [],
            subquery_docs: str = ""
    ):
        ## context placeholder
        context = "\n"

        ## unpack documents for context
        if len(documents) > 0:
            for rank, doc in enumerate(documents, start=1):
                context += f"Relevant Document: {rank}::\n" + doc.page_content + "\n"

        ## llm call for generating the response
        prompt = ChatPromptTemplate.from_template(self.rag_prompt_template)
        chain = prompt | self.model | StrOutputParser()
        resp = chain.invoke({
            "chat_history": memory,
            "context": context,
            "multi_hop_context": subquery_docs
        })

        return resp