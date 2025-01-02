from fastapi import APIRouter, HTTPException, status
from models import ImageRequest
from services import compare_face_service

router = APIRouter()
async def compare_face(request: ImageRequest):
    try:
        result = await compare_face_service(request)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
