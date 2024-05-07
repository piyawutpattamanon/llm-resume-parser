import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.chains.extractors.base import Base
from src.utils.extract_json import extract_json_only


class Job(BaseModel):
    title: str|None = Field(description="The title of the job position")
    employee: str|None = Field(description="The company name of the job position")
    start_date: str|None = Field(description="The start date of the job position")
    end_date: str|None = Field(description="The end date of the job position")
    city: str|None = Field(description="The city of the job position", default=None)
    country: str|None = Field(description="The country of the job position", default=None)
    country_code: str|None = Field(description="The country code of the job position", default=None)


class List_of_jobs(BaseModel):
    jobs: List[Job]



class JobExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    # def validate(self, text):
    #     result = extract_json_only(text)
    #     if len(result) == 0:
    #         raise ValueError("No job position found in the resume")
    #     return result[0]
    
    def get_fallback_value(self, text):
        return '{"error": "this is a fixed value"}'
    
    def get_prompt_template(self):
        parser = PydanticOutputParser(pydantic_object=List_of_jobs)

        format_instructions = parser.get_format_instructions()

        prompt_text = """
you're a resume parser.
in his work experience, what are his job positions and these attributes related to the position: title, co, end_date, start_date, city, country_code.
no more attributes other than this.

{format_instructions}
====
below is the resume
====
{resume_content}
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
            bound=minprompt,
            max_attempt_number=2,
        ).with_fallbacks(
            [RunnableLambda(self.get_fallback_value)]
        )

        return safe_min


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

    from src.llms.langchain_lmstudio import LMStudioLLM

    llm = LMStudioLLM(n=10)

    from langchain.globals import set_debug
    set_debug(True)

    extractor = JobExtractor(llm=llm)
    chain = extractor.get_chain()

    

    resume_content = resumes[102]['content']

    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.split_text(resume_content)
    print('number of tokens', len(chunks))

    import datetime
    start_time = datetime.datetime.now()


    result = chain.invoke({'resume_content': resume_content})

    # print('=== clean result ===')
    # print(json.dumps(json.loads(result), indent=4))
    print('=== raw result ===')
    print(result)

    end_time = datetime.datetime.now()
    print(f'Time taken: {end_time - start_time}')