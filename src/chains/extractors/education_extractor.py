import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.chains.extractors.base import Base


class Education(BaseModel):
    city: str | None = Field(description="The city of the education", default=None)
    school: str | None = Field(description="The school name")
    country: str | None = Field(
        description="The country of the education", default=None
    )
    end_date: str | None = Field(description="The end date of the education")
    start_date: str | None = Field(description="The start date of the education")
    degree_name: str | None = Field(description="The name of the degree", default=None)
    description: str | None = Field(
        description="Additional description of the education", default=None
    )
    country_code: str | None = Field(
        description="The country code of the education", default=None
    )
    degree_major: str | None = Field(
        description="The major of the degree", default=None
    )


class EducationExperience(BaseModel):
    education: List[Education]


class EducationExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def get_fallback_value(self, text):
        return {
            "education": [],
            "education_error": True,
        }

    def get_prompt_template(self):
        parser = PydanticOutputParser(pydantic_object=EducationExperience)

        format_instructions = parser.get_format_instructions()

        prompt_text = """
this is a resume
=====
{resume_content}
=====
you are resume parser.
extract his education as specified in his resume.
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
        ).with_fallbacks([RunnableLambda(self.get_fallback_value)])

        formatted_chain = safe_min

        return formatted_chain

    def get_chain(self):
        if self.chain is None:
            self.chain = self.build_chain()
        return self.chain
