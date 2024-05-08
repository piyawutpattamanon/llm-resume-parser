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
