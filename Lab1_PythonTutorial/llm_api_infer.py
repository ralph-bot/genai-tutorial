import requests

# How many person have the openai api key or deepseek api key ? 
url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-V3.2-Exp",
    "messages": [
        {
            "role": "user",
            "content": "当前最好的大模型是哪个？"
            #"content": "What opportunities and challenges will the Chinese large model industry face in 2025? "
        }
    ],
    "stream": False,
    "max_tokens": 512,
    "min_p": 0.05,
    "stop": None,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
    "tools": [
        {
            "type": "function",
            "function": {
                "description": "<string>",
                "name": "<string>",
                "parameters": {},
                "strict": False
            }
        }
    ]
}
headers = {
    "Authorization": "Bearer xxx",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)