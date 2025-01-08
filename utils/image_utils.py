import base64
from fastapi import HTTPException ,status
import numpy as np
import cv2

import base64
import cv2
import numpy as np
from fastapi import HTTPException, status

def decode_base64_image(base64_str: str) -> np.ndarray:
    try:
        
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
       
        # Decode base64 string
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        
        # Decode the image using OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Cannot decode image")

        # Ensure the image is in 8-bit RGB format
        if len(img.shape) == 2:
            # Grayscale image: Convert to 3-channel RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # 4-channel image (e.g., PNG with alpha): Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Validate image type
        if img.dtype != np.uint8:
            raise ValueError("Image must be 8-bit")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"รูปภาพไม่ถูกต้อง: {str(e)}"
        )
# def decode_base64_image(base64_str):
#     try:
#         # หาก base64 มีส่วนของ data URL, ตัดออก
#         if base64_str.startswith("data:image/"):
#             base64_str = base64_str.split(",")[1]
        
#         # แปลง base64 เป็นข้อมูล byte
#         img_data = base64.b64decode(base64_str)
#         # แปลงข้อมูล byte เป็น numpy array
#         img_array = np.frombuffer(img_data, dtype=np.uint8)
#         # ใช้ cv2.imdecode เพื่อแปลง numpy array เป็นภาพ
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # โหลดเป็นสี (BGR)
        
#         if img is None:
#             raise ValueError("ไม่สามารถแปลง Base64 เป็นภาพได้")
        
#         # ตรวจสอบว่าเป็นภาพ RGB หรือ grayscale ที่รองรับ
#         if len(img.shape) == 2:  # ถ้าเป็น grayscale
#             try:
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             except Exception as e:
#                 raise ValueError(f"ไม่สามารถแปลงภาพ grayscale เป็น RGB: {str(e)}")
#         elif len(img.shape) == 3:  # ถ้าเป็น RGB
#             img_rgb = img
#         else:
#             raise ValueError("รูปภาพต้องมี 1 หรือ 3 ช่องข้อมูล")
        
#         return img_rgb
#     except Exception as e:
#         raise ValueError(f"ข้อผิดพลาดในการแปลง Base64 เป็นภาพ: {str(e)}")
