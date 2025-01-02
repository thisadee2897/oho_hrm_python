import base64
import numpy as np
import cv2

def decode_base64_image(base64_str):
    try:
        if base64_str.startswith("data:image/"):
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        img = np.array(bytearray(img_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("ไม่สามารถแปลง Base64 เป็นภาพได้")

        return img
    except Exception as e:
        raise ValueError(f"ข้อผิดพลาดในการแปลง Base64 เป็นภาพ: {str(e)}")
