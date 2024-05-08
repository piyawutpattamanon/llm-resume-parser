# LLM Resume Parser

This is a PDF file resume parser that uses LLM as the extraction engine.
The system performance is optimized some what utilizing asynchronous programming.

## Setup

```sh
pip install -r requirements.txt
```


## Running API Server

This demo uses Perplexity AI as the foundation LLM. We need to set the API key as an environment variable.
```sh
export PPLX_API_KEY=pplx-xxxxxxxxx-xxxxxxxxx-xxxxxxxxx-xxxxxxxxx
```

After that, we can run the API server with the following command.
```sh
python -m uvicorn src.api.main:app
```

## Playing with the UI
The API server provide a form upload that can be access at this default URL
```url
http://localhost:8000/
```
You can upload a pdf resume file. Wait a few seconds and you will see JSON response of the extracted data from the resume.


## Performance

- Response Time: 6-12 seconds
- Evaluation Metrics: Not implemented yet


## Unit Test
```sh
pytest
```