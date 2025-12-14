from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI()

origins = ["https://mufasa-real-assistant.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    portal_id: str
    resume_code: str | None = None

@app.post("/ask")
async def ask(req: AskRequest):
    prompt = f"[PORTAL_ID={req.portal_id}] [RESUME={req.resume_code}] {req.question}"
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are Mufasa Real Assistant, a guide using Afrocentric, unpolarized knowledge schemas."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message["content"]
    # simple next-code mock
    next_code = f"{req.portal_id}_NEXT"
    return {"answer": answer, "next_resume_code": next_code}

@app.post("/media/presentation")
async def media_presentation(data: dict):
    # placeholder for future multimedia output
    return {"status": "ok", "slides": []}
