import json
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables.retry import RunnableRetry
from langchain.prompts.chat import ChatPromptTemplate
from src.chains.extractors.base import Base
from src.utils.extract_json import extract_json_only
class JobExtractor(Base):
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
this is a job resume
====
{resume_content}
====
extract his job positions and these attributes related to the position: title, employer, end_date, start_date, city, country_code.
no more attributes other than this.
output in json.

example output
[
    {{
      "city": "",
      "job_title": "NCAT ASEC",
      "country": "",
      "company": "Hickman Property Holdings LLC",
      "end_date": "2023-",
      "start_date": "2022-2",
      "country_code": "IN"
    }}
]


if there is no job position in his resume. output a blank array like this
[ ]"""
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

    extractor = JobExtractor(llm=llm)
    chain = extractor.get_chain()

    result = chain.invoke({'resume_content': resumes[0]['content']})

    print('=== clean result ===')
    print(json.dumps(json.loads(result), indent=4))
    print('=== raw result ===')
    print(result)