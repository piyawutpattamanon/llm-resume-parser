# pure LM studio. not integrated with langchain

import requests
import json

def do_llm_completion(messages):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        # "model": "LM Studio Community/Phi-3-mini-4k-instruct-GGUF",
        "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": -1,
        "stream": True
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    if response.status_code == 200:
        content = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                chunk_line = chunk.decode('utf-8')
                if chunk_line.startswith('data:'):
                    chunk_content = chunk_line[len('data: '):]
                    if not chunk_content.startswith('[DONE]'):
                        # print(chunk_content)
                        chunk_data = json.loads(chunk_content)
                        if chunk_data['choices'][0]['finish_reason'] is None and chunk_data.get('id'):
                            # print(chunk_data)
                            # print('====', chunk_data['choices'][0]['delta']['role'], '====')
                            # print(chunk_data['choices'][0]['delta']['content'])
                            content += chunk_data['choices'][0]['delta']['content']
        return content
    else:
        print(f"Request failed with status code: {response.status_code}")

    return None