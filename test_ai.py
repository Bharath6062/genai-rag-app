from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say OK and explain what Python is in one sentence."}
    ],
)

print(resp.choices[0].message.content)
