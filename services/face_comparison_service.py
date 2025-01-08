import cv2
import face_recognition as face
import numpy as np
from fastapi import HTTPException, status 
import os
import requests as rq
from models.check_in_model import ImageRequest
from utils.image_utils import decode_base64_image

async def compare_face_service(request : ImageRequest):
    try:
        # แปลง base64 ไปเป็นภาพ
        img = decode_base64_image(request.check_in_image)
        if len(img.shape) < 3:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รูปภาพต้องมี 3 ช่องข้อมูล (RGB หรือ BGR)")
        if img.dtype != np.uint8:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รูปภาพต้องเป็นชนิดข้อมูล 8-bit")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # ตรวจสอบใบหน้าในภาพ
        face_locations = face.face_locations(img, model="hog")
        if len(face_locations) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่พบใบหน้าในภาพที่ส่งมา")
        # ใบหน้าที่ต้องการเปรียบเทียบ
        try:
            response = rq.get(request.profile_image)
            response.raise_for_status()  # ตรวจสอบว่าโหลดสำเร็จหรือไม่
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            profile_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            profile_image = cv2.cvtColor(profile_image, cv2.COLOR_BGR2GRAY)
            profile_image = cv2.cvtColor(profile_image, cv2.COLOR_GRAY2BGR)
            profile_face_encoding = face.face_encodings(profile_image)
            if len(profile_face_encoding) == 0:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่พบใบหน้าจากภาพที่เก็บไว้")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"เกิดข้อผิดพลาดในการโหลดภาพจากเซิร์ฟเวอร์: {str(e)}")
        profile_face_encoding = profile_face_encoding[0]
        # ค้นหาตำแหน่งใบหน้าและเอกลักษณ์ใบหน้าในภาพที่ส่งเข้ามา
        face_encodings = face.face_encodings(img, face_locations)
        # return {"result": True, "distance": 0, "message": "พบใบหน้าที่ตรงกัน"}
        # ตรวจสอบใบหน้าที่พบ
        for face_encoding in face_encodings:
            face_distances = face.face_distance([profile_face_encoding], face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:  # ใช้ค่าความคล้ายที่ต่ำกว่า 0.6
                return {"result": True, "distance": face_distances[best_match_index], "message": "พบใบหน้าที่ตรงกัน"}

        return {"result": False, "distance": face_distances[best_match_index], "message": "ไม่พบใบหน้าที่ตรงกัน"}
    except HTTPException as http_exc:
        raise http_exc
