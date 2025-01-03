from fastapi import APIRouter, HTTPException, UploadFile, status
from models import ImageRequest
from services import compare_face_service

router = APIRouter()
@router.post("/face/")
async def compare_face(request: ImageRequest, file: UploadFile):
    try:
        result = await compare_face_service(request, file)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
