from pydantic import BaseModel
class ImageRequest(BaseModel):
    profile_image: str

class ImageResponse(BaseModel):
    match: bool | None
    persent: float | None
    message: str | None