from fastapi import FastAPI
from api import compare_face
from models import ImageRequest
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

@app.post("/compare-face/")
async def compare_face_endpoint(request: ImageRequest):
    print(request.profile_image)
    return await compare_face(request)
