import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generation import ask_question  # Import the function from generation.py

app = FastAPI()

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="Which is the study that is related to Heart and Vascular ?"
    )

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask_question", response_model=QueryResponse, summary="Ask a question", description="This endpoint accepts a question and returns an answer.", response_description="The answer to the question.")
async def ask(query: QueryRequest):
    """
    Example usage:
    - Sample question 1: Which is the study that is related to Heart and Vascular ?
    - Sample question 2: Any information on ?
    - Sample question 3:?
    """
    try:
        answer = ask_question(query.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

   
