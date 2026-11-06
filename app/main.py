
from dotenv import load_dotenv
import os

load_dotenv()

# Prompt Template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following into {language}:"),
        ("user", "{text}")
    ]
)

# Groq LLM
from langchain_groq import ChatGroq
import os

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from environment variables")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0
)

# Output Parser
from langchain_core.output_parsers import StrOutputParser

output = StrOutputParser()

# Create Chain
chain = prompt | llm | output

# FastAPI and LangServe
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="Simple LangServe Translation API",
    version="1.0",
    description="Translation API powered by Groq"
)

add_routes(app, chain, path="/chain")

@app.get("/")
async def health_check():
    return {
        "message": "LangServe API with Groq is running on Vercel!"
    }
