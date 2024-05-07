import json
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables.retry import RunnableRetry
from langchain.prompts.chat import ChatPromptTemplate
from src.chains.extractors.base import Base
from src.utils.extract_json import extract_json_only
class EducationExtractor(Base):
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def validate(self, text):
        result = extract_json_only(text)
        if len(result) == 0:
            raise ValueError("No job position found in the resume")
        return result[0]
    
    def get_fallback_value(self, text):
        return "this is a fixed value"
    
    def get_prompt_template(self):
        prompt_text = """
this is a resume
=====
{resume_content}
=====
you are resume parser.
extract his education as specified in his resume.
output in json

example output
[
  {{
    "city": "",
    "school": "State University",
    "country": "",
    "end_date": "2024-5",
    "start_date": "-",
    "degree_name": "",
    "description": "",
    "country_code": "NC",
    "degree_major": ""
  }},
  {{
    "city": "",
    "school": "Central Piedmont Community College",
    "country": "",
    "end_date": "2021-12",
    "start_date": "2020-8",
    "degree_name": "",
    "description": "North Carolina Agricultural and Technical State University, Greensboro, NC\nMay 2024\nGPA: 3.4\nCentral Piedmont Community College, Charlotte NC\nAugust 2020 - December 2021\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023",
    "country_code": "NC",
    "degree_major": ""
  }}
]

if there is no education in his resume, output a blank array like this
[ ]
    """
        prompt = ChatPromptTemplate.from_template(prompt_text)
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

    extractor = EducationExtractor(llm=llm)
    chain = extractor.get_chain()

    result = chain.invoke({'resume_content': resumes[0]['content']})

    print('=== clean result ===')
    print(json.dumps(json.loads(result), indent=4))
    print('=== raw result ===')
    print(result)