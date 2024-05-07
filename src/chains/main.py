import json
from typing import List
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import TokenTextSplitter

from src.chains.extractors.education_extractor import EducationExtractor
from src.chains.extractors.job_extractor import JobExtractor
from src.chains.extractors.skill_extractor import SkillExtractor
from src.chains.extractors.person_extractor import PersonExtractor


class ChunkMapper:
    def __init__(self, chunk_index: int):
        self.chunk_index = chunk_index

    def __call__(self, data) -> dict:
        return {"resume_content": data["resume_chunks"][self.chunk_index]}


class ResumeParserChain:
    def __init__(self, llm):
        self.llm = llm
        self.chain = None

    def split_chunks(self, resume_content) -> List[str]:
        text_splitter = TokenTextSplitter(chunk_size=10000, chunk_overlap=2000)
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
                    "education": education_chain,
                    "experience": job_chain,
                    "skills": skill_chain,
                    "personal": person_chain,
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

    async def parse(self, resume_content):
        resume_chunks = self.split_chunks(resume_content)

        # print('number of chunks', len(resume_chunks))
        # input('press to continue')

        single_chunk_chain = self.get_single_chunk_chain()

        combined_chain = RunnableParallel(
            {
                chunk_index: ChunkMapper(chunk_index) | single_chunk_chain
                for chunk_index in range(len(resume_chunks))
            }
        )

        def combine_chunk_results(raw):

            result = {}

            info = {}
            for chunk_index in raw:
                for k in raw[chunk_index]["personal"]:
                    if raw[chunk_index]["personal"][k]:
                        info[k] = raw[chunk_index]["personal"][k]
                    else:
                        info[k] = info.get(k)
            result["personal"] = info

            info = []
            for chunk_index in raw:
                info += raw[chunk_index]["experience"]
            result["experience"] = info

            info = []
            for chunk_index in raw:
                info += raw[chunk_index]["education"]
            result["education"] = info

            info = []
            for chunk_index in raw:
                info += raw[chunk_index]["skills"]
            result["skills"] = info

            return result

        combined_chain = combined_chain | RunnableLambda(combine_chunk_results)

        # from langchain.chains.graphviz import render_chain
        # render_chain(combined_chain, "chain_diagram.png", vertical=True)

        config = RunnableConfig(max_concurrency=10)

        return await combined_chain.ainvoke(
            {
                "resume_chunks": resume_chunks,
            },
            config=config,
        )


if __name__ == "__main__":
    resumes = []
    with open("Entity Recognition in Resumes.jsonl") as f:
        for line in f:
            resume = json.loads(line)
            resumes.append(resume)

    from src.llms.langchain_lmstudio import LMStudioLLM

    llm = LMStudioLLM(n=10)

    from langchain_community.chat_models import ChatPerplexity

    # model_name = 'llama-3-sonar-small-32k-online'
    # model_name = 'pplx-70b-online'
    model_name = "llama-3-sonar-large-32k-online"
    pplx_llm = ChatPerplexity(temperature=0.1, model=model_name)

    # from langchain.llms import PerplexityAI
    # from langchain_community.llms import PerplexityAI
    # pplx_llm = PerplexityAI(model=model_name, temperature=0.1)

    from langchain_anthropic import ChatAnthropic

    claude_llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        # model="claude-3-opus-20240229",
        # model="claude-3-sonnet-20240229",
        temperature=0.2,
        # max_tokens=1024,
    )

    from langchain.globals import set_debug

    set_debug(True)

    resume_paser = ResumeParserChain(llm=pplx_llm)

    import asyncio

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(resume_paser.parse(resumes[102]["content"]))

    # Visualize the workflow
    # import langchain_visualizer
    # langchain_visualizer.visualize(resume_paser.parse, resume_content=resumes[102]['content'])
    # result = resume_paser.parse(resumes[102]['content'])

    # print('=== raw result ===')
    # print(result)
    # for k in result.keys():
    #     result[k] = json.loads(result[k])
    print("=== clean result ===")
    print(json.dumps(result, indent=4))
