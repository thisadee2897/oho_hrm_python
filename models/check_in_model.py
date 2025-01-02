from pydantic import BaseModel
from typing import Optional
class ImageRequest(BaseModel):
    profile_image: str 
    check_in_image: str

class ImageResponse(BaseModel):
    match: bool
    distance: float
    message: str