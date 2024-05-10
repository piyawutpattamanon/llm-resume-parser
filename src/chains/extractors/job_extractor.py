import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.chains.extractors.base import Base


class Job(BaseModel):
    title: str | None = Field(description="The title of the job position")
    employer: str | None = Field(description="The company name of the job position")
    start_date: str | None = Field(description="The start date of the job position")
    end_date: str | None = Field(description="The end date of the job position")
    city: str | None = Field(description="The city of the job position", default=None)
    country: str | None = Field(
        description="The country of the job position", default=None
    )
    country_code: str | None = Field(
        description="The country code of the job position", default=None
    )


class JobExperience(BaseModel):
    experience: List[Job]


class JobExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def get_fallback_value(self, text):
        return {
            "experience": [],
            "experience_error": True,
        }

    def get_prompt_template(self):
        parser = PydanticOutputParser(pydantic_object=JobExperience)

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

        formatter = RunnableLambda(lambda result: result.dict())

        safe_min = RunnableRetry(
            bound=minprompt | formatter,
            max_attempt_number=3,
        ).with_fallbacks([RunnableLambda(self.get_fallback_value)])

        formatted_chain = safe_min

        return formatted_chain

    def get_chain(self):
        if self.chain is None:
            self.chain = self.build_chain()
        return self.chain
