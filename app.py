from fastapi import FastAPI
from pydantic import BaseModel
from progress_simple_chatbox import generate_response

app = FastAPI()

class RequestBody(BaseModel):
    description: str

@app.post("/chatbox")
async def predict(item: RequestBody):
    description = item.description
    response = generate_response(description)
    return {
        "code": 200,
        "message": "success",
        "response": response
    }