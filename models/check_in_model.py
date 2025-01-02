from pydantic import BaseModel
from typing import Optional
class ImageRequest(BaseModel):
    profile_image: Optional[str] = None  # path ของภาพที่เก็บไว้ในเซิร์ฟเวอร์
    check_in_image: str  # รหัส base64 ของภาพ
