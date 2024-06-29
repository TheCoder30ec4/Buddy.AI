from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import chat_with_bot  # Ensure this is correctly imported

app = FastAPI()

# CORS settings
origins = [
    "http://localhost:3000",  # Replace with your frontend URL if necessary
    "http://127.0.0.1:3000",  # Replace with your frontend URL if necessary
    "http://localhost:5173",  # Add this line
    "http://127.0.0.1:5173"   # Add this line
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the incoming data
class TurtleData(BaseModel):
    Data: str

@app.get('/')
async def root():
    """
    Root endpoint for testing.
    """
    return {'message': 'Welcome to the Turtle API!'}

@app.post('/input')
async def input_prompt(turtle: TurtleData):
    """
    Endpoint to receive data and respond with a message.
    
    Args:
        turtle (TurtleData): The data payload sent in the request body.
    
    Returns:
        dict: A confirmation message with the received data.
    """
    if not turtle.Data:
        raise HTTPException(status_code=400, detail="Data field is required")
    
    response = chat_with_bot(turtle.Data)  # Assuming chat_with_bot returns a string or dict
    return {'message': response}
