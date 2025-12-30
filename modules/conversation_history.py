from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Union

## class for conversation memory
class ConversationSummaryMemory:
    def __init__(self, model, k: int) -> None:
        self.conversations = []
        self.model = model
        self.window = k*2
        self._summary_cache = ""
    
    ## windowed conversation list
    @property
    def windowed_conversation(self):
        if len(self.conversations) > self.window:
            return self.conversations[-self.window:]
        return self.conversations

    ## method to pretty print
    @staticmethod
    def __pretty_print(conversations) -> str:
        entire_conversation = []
        for turn in conversations:
            entire_conversation.append(f"> {turn.type}: {turn.content}")
        return "\n".join(entire_conversation)

    ## private method for initiating a langchain chain
    def __initiate_chain(self):
        template = """
        You are an expert in summarizing conversations between a human and an AI chatbot. Your task is to write a short, clear summary that keeps the main facts, tone, and context of the conversation intact while removing unnecessary or repetitive parts. The summary should be concise but informative enough for another system to understand what the conversation was about and its overall direction.

        Input Conversation:
        {conversation}

        Output:
        A brief, well-written paragraph summarizing the conversation.
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["conversation"]
        )
        chain = prompt | self.model | StrOutputParser()
        return chain
    
    ## method to add conversation one-by-one
    def append(self, conversation: Union[AIMessage, HumanMessage]) -> None:
        self.conversations.append(conversation)
        if len(self.conversations) > self.window:
            prev = self._summary_cache
            total_conversation_context = (
                "Previous Conversation Summary: " + prev
                + "\nLatest Turns:\n" + self.__pretty_print(self.windowed_conversation)
            )
            chain = self.__initiate_chain()
            self._summary_cache = chain.invoke({"conversation": total_conversation_context})
        # else:
        #     total_conversation_context = self.__pretty_print(self.conversations)

    ## generate summary
    @property
    def summary(self) -> str:
        return self._summary_cache
    
    ## object print
    def __str__(self) -> str:
      if len(self.conversations) > self.window:
        return f"Last {self.window} turns of the Conversation:\n\n{self.__pretty_print(self.windowed_conversation)}\n\nPrevious Conversation Summary: {self.summary}"
      return f"Latest Conversation:\n\n{self.__pretty_print(self.windowed_conversation)}"
    
## ---- TEST ----
if __name__=="__main__":
   from langchain_cohere import ChatCohere

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
       AIMessage(content="Yes, he worked in Tropic Thunder with Tom Cruise."),
       HumanMessage(content="Who is he married to?")
   ]

   ## instantiate memory
   memory = ConversationSummaryMemory(
       model=llm,
       k=2
   )

   ## add dummy messages to the memory
   for message in messages:
       memory.append(message)

   print("Chat History".center(100, "-"))
   print(memory)

   # """
   # (rag-based-legal-assistant) C:\Users\sodey\Downloads\_projects\RAG-based-Legal-Assistant>uv run -m modules.conversation_history
   #  --------------------------------------------Chat History--------------------------------------------
   #  Last 4 turns of the Conversation:
   #
   #  > ai: His best works include Chaplin, Sherlock Holmes, The Judge, Oppenheimer and many more.
   #  > human: Has he ever acted in a movie along with Tom Cruise?
   #  > ai: Yes, he worked in Tropic Thunder with Tom Cruise.
   #  > human: Who is he married to?
   #
   #  Previous Conversation Summary: The conversation centers on Robert Downey Jr.'s career, starting with confirmation of his iconic role as Iron Man in the MCU. The discussion highlights his acclaimed performances in films like *Chaplin*, *Sherlock Holmes*, *The Judge*, and *Oppenheimer*. The human inquires about potential collaborations with Tom Cruise, leading to the revelation that Downey and Cruise co-starred in *Tropic Thunder*. The conversation then shifts to a personal question about Downey's marital status, indicating a broader interest in both his professional and personal life.
   # """