from fastapi import FastAPI , HTTPException, Request
import uvicorn
from api import compare_face
from models import ImageRequest
app = FastAPI()

MAX_BODY_SIZE = 50 * 1024 * 1024
@app.middleware("http")
async def limit_body_size_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    print(content_length)
    if content_length and int(content_length) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large")
    return await call_next(request)

@app.get("/")
async def root():
    return HTTPException(status_code=200, detail="Hello World")


@app.post("/compare-face/")
async def compare_face_endpoint(request: ImageRequest):
    return await compare_face(request)

if __name__ == "__main__":
    uvicorn.run("oho_hrm:app", host="localhost", port=9898,
        log_level="info",
        limit_concurrency=100,
        limit_max_requests=1000,
        limit_max_request_size=50 * 1024 * 1024
        )
    
    
