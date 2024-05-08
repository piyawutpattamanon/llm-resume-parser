# want fastapi application
# receive a pdf file upload
# endpoint A : extract the content into text using langchain
# endpoint B : pdf file upload form

import fastapi
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.document_loaders import PyPDFLoader
import tempfile
import os

from langchain_community.chat_models import ChatPerplexity

from src.chains.main import ResumeParserChain

from langchain.globals import set_debug

set_debug(True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: fastapi.Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def extract_uploaded_pdf_content(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Save uploaded file to a temporary file
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        os.unlink(temp_file_path)

    text = ""
    for page in pages:
        text += page.page_content
    return text


@app.post("/v1.0/extract_text")
async def extract_text(file: UploadFile = File(...)):
    resume_content = await extract_uploaded_pdf_content(file)

    model_name = os.environ.get("PPLX_MODEL_NAME", "llama-3-sonar-large-32k-online")
    pplx_llm = ChatPerplexity(temperature=0.1, model=model_name)

    resume_parser = ResumeParserChain(llm=pplx_llm)
    result = await resume_parser.parse(resume_content)

    return result

    # return {"text": text}
