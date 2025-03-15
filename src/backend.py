# FastAPI Backend for Bot Detection
from fastapi import FastAPI
app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Bot Detection API'}