from pydantic import BaseModel
from typing import Optional
class ImageRequest(BaseModel):
    profile_image: str

class ImageResponse(BaseModel):
    match: bool
    persent: float
    message: str