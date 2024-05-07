import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables.retry import RunnableRetry
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.chains.extractors.base import Base

class Person(BaseModel):
    gender: str|None = Field(description="The gender of the person")
    birthplace: str|None = Field(description="The birthplace of the person")
    first_name: str|None = Field(description="The first name of the person")
    family_name: str|None = Field(description="The family name of the person")
    middle_name: str|None = Field(description="The middle name of the person")
    nationality: str|None = Field(description="The nationality of the person")
    date_of_birth: str|None = Field(description="The date of birth of the person")
    marital_status: str|None = Field(description="The marital status of the person")
    country_code: str|None = Field(description="The country code of the person")


class PersonExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None
    
    def get_fallback_value(self, text):
        return '{"error": "this is a fixed value"}'
    
    def get_prompt_template(self):
        parser = PydanticOutputParser(pydantic_object=Person)

        format_instructions = parser.get_format_instructions()

        prompt_text = """
this is a resume
=====
{resume_content}
=====
what's the name of this person who is the owner of this resume?
what country he/she is from? is he/she married? should he/she be male or female?
output in json. no explanation. no comment

{format_instructions}
"""
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["resume_content"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def build_chain(self):
        minprompt = self.get_prompt_template()

        safe_min = RunnableRetry(
            bound=minprompt | RunnableLambda(lambda result: result.dict()),
            max_attempt_number=2,
        ).with_fallbacks(
            [RunnableLambda(self.get_fallback_value)]
        )

        formatted_chain = safe_min

        return formatted_chain

    def get_chain(self):
        if self.chain is None:
            self.chain = self.build_chain()
        return self.chain
    
if __name__ == '__main__':
    resumes = []
    with open('Entity Recognition in Resumes.jsonl') as f:
        for line in f:
            resume = json.loads(line)
            resumes.append(resume)

    import datetime

    start_time = datetime.datetime.now()

    from src.llms.langchain_lmstudio import LMStudioLLM

    llm = LMStudioLLM(n=10)

    from langchain.globals import set_debug
    set_debug(True)

    extractor = PersonExtractor(llm=llm)
    chain = extractor.get_chain()

    result = chain.invoke({'resume_content': resumes[0]['content']})

    print('=== clean result ===')
    print(json.dumps(result, indent=4))
    # print('=== raw result ===')
    # print(result)

    end_time = datetime.datetime.now()
    print(f'Time taken: {end_time - start_time}')