import json
from typing import List
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_core.runnables.retry import RunnableRetry
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import TokenTextSplitter

from src.chains.extractors.education_extractor import EducationExtractor
from src.chains.extractors.job_extractor import JobExtractor
from src.chains.extractors.skill_extractor import SkillExtractor
from src.chains.extractors.person_extractor import PersonExtractor


class ResumeParserChain:
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def split_chunks(self, resume_content) -> List[str]:
        text_splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=500)
        chunks = text_splitter.split_text(resume_content)

        return chunks

    def get_single_chunk_chain(self):
        if self.chain is None:

            education_chain = EducationExtractor(llm=self.llm).get_chain()
            job_chain = JobExtractor(llm=self.llm).get_chain()
            skill_chain = SkillExtractor(llm=self.llm).get_chain()
            person_chain = PersonExtractor(llm=self.llm).get_chain()
            
            chain = RunnableParallel(
                {
                "education":education_chain,
                "experience":job_chain,
                "skills":skill_chain,
                "personal":person_chain,
                }
            )

            def combine_fields(src: dict):
                result = {}
                for k in src:
                    result.update(src.get(k))
                return result
            
            chain = chain | RunnableLambda(combine_fields)

            self.chain = chain
        
        return self.chain


    def parse(self, resume_content):
        chunks = self.split_chunks(resume_content)

        print('number of chunks', len(chunks))
        input('press to continue')

        single_chunk_chain = self.get_single_chunk_chain()

        combined_chain = RunnableParallel(
            {
                f'chunk_{chunk_index}': single_chunk_chain
                for chunk_index
                in range(len(chunks))
            }
        )

        # def combine_chunks(src):


        

        return combined_chain.invoke({
            "resume_content": resume_content
        })
    
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

    resume_paser = ResumeParserChain(llm=llm)
    result = resume_paser.parse(resumes[0]['content'])

    # print('=== raw result ===')
    # print(result)
    # for k in result.keys():
    #     result[k] = json.loads(result[k])
    print('=== clean result ===')
    print(json.dumps(result, indent=4))
