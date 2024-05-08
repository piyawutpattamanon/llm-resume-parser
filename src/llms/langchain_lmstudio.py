from typing import Any, List, Mapping, Optional
import requests
import json
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

# Change the Url according to the Server
URL = "http://localhost:1234/v1/chat/completions"


class LMStudioLLM(LLM):
    n: int
    url = URL

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500,
                "stream": False,
            }
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", self.url, headers=headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
