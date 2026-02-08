import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
# ----------------------------
# Azure OpenAI client
# ----------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("MODEL_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))
print(os.getenv("MODEL_VERSION"))
print(DEPLOYMENT_NAME)



SYSTEM_PROMPT = """
You are a search optimization agent.

GOAL:
Improve retrieval quality by ADDING missing specific keywords or constraints.

RULES:
- Do NOT paraphrase the query
- Do NOT remove existing keywords
- ONLY add missing specific terms
- If already optimal, return NO_CHANGE
- Output ONLY the optimized query or NO_CHANGE
"""

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

class Prompt(BaseModel):
    query: str

@app.post("/optimize")
def optimize_query(data: Prompt):
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Original query:\n{data.query}"}
        ],
        temperature=0.1,
        max_tokens=100,
    )

    return {
        "optimized_query": response.choices[0].message.content.strip()
    }

@app.get("/health")
def health():
    return {"status": "ok"}

