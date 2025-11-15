from openai import OpenAI
from dotenv import load_dotenv
import os

# load .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("Loaded API key:", "YES" if api_key else "NO")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print("API key is working ✔")
    print("Response:", response.choices[0].message["content"])

except Exception as e:
    print("API key is NOT working ❌")
    print("Error:", e)
