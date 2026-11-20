
import requests

url = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2:3b",
    "messages": [
        {
            "role": "system",
            "content": "You are TURBO, my personal AI assistant. Be helpful and concise."
        },
        {
            "role": "user",
            "content": "Introduce yourself in one sentence."
        }
    ],
    "stream": False
}

try:
    response = requests.post(url, json=data, timeout=120)
    response.raise_for_status()

    result = response.json()
    print("TURBO:", result["message"]["content"])

except requests.exceptions.ConnectionError:
    print("Could not connect to Ollama. Make sure Ollama is running.")

except requests.exceptions.RequestException as error:
    print("Request failed:", error)
