import json
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables.retry import RunnableRetry
from langchain.prompts.chat import ChatPromptTemplate
from src.chains.extractors.base import Base
class JobExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def validate(self, text):
        print('==== COME TO THIS ====')
        print(text)
        print('.......')
        json.loads(text)
        return text
    
    def get_fallback_value(self, text):
        return "this is a fixed value"
    
    def get_prompt_template(self):
        prompt = ChatPromptTemplate.from_template("from {text}. what's the minimal number? answer in json only. example {{ \"min\" : 999 }}")
        return prompt

    def build_chain(self):
        minprompt = self.get_prompt_template()

        safe_min = RunnableRetry(
            bound=minprompt | self.llm | RunnableLambda(self.validate),
            max_attempt_number=2,
        ).with_fallbacks(
            [RunnableLambda(self.get_fallback_value)]
        )

        return safe_min


    def get_chain(self):
        if self.chain is None:
            self.chain = self.build_chain()
        return self.chain