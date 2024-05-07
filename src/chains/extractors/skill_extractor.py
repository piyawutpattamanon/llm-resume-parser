import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.chains.extractors.base import Base


class Skill(BaseModel):
    name: str = Field(description="The name of the skill")
    type: str | None = Field(description="The type of the skill (hard or soft)")


class SkillSet(BaseModel):
    skills: List[Skill]


class SkillExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def get_fallback_value(self, text):
        return {
            "skills": [],
            "skills_error": True,
        }

    def get_prompt_template(self):
        parser = PydanticOutputParser(pydantic_object=SkillSet)

        format_instructions = parser.get_format_instructions()

        prompt_text = """
this is a resume
=====
{resume_content}
=====
you are resume parser.
extract his skills as specified in his resume. both hard and soft skills.

{format_instructions}

if there is no skills in his resume, output a blank array like this
{{ "skills": [] }}
"""
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["resume_content"],
            partial_variables={"format_instructions": format_instructions},
        )
        return prompt | self.llm | parser

    def build_chain(self):
        minprompt = self.get_prompt_template()

        def formatting(item: SkillSet):
            formatted = {
                "skills": [skill["name"] for skill in item.model_dump()["skills"]]
            }

            return formatted

        safe_min = RunnableRetry(
            bound=minprompt | RunnableLambda(formatting),
            max_attempt_number=2,
        ).with_fallbacks([RunnableLambda(self.get_fallback_value)])

        formatted_chain = safe_min

        return formatted_chain

    def get_chain(self):
        if self.chain is None:
            self.chain = self.build_chain()
        return self.chain
