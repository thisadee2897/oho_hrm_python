import face_recognition as face
import numpy as np
from utils import decode_base64_image
from fastapi import HTTPException, status
from models import ImageRequest, ImageResponse
import os

async def load_profile_image(profile_image_path: str):
    try:
        if not os.path.exists(profile_image_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบไฟล์ภาพในเซิร์ฟเวอร์")
        
        profile_image = face.load_image_file(profile_image_path)
        profile_face_encoding = face.face_encodings(profile_image)
        
        if len(profile_face_encoding) == 0:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่พบใบหน้าจากภาพที่เก็บไว้")
        
        return profile_face_encoding[0]
    
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการโหลดภาพจากเซิร์ฟเวอร์")

async def compare_face_service(request: ImageRequest):
    try:
        # แปลง base64 เป็นภาพ
        img = decode_base64_image(request.check_in_image)
        
        # ตรวจสอบการมีใบหน้าในภาพที่ส่งมา
        face_locations = face.face_locations(img, model="hog")
        if len(face_locations) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่พบใบหน้าในภาพที่ส่งมา")
        
        # โหลดข้อมูลใบหน้าจากภาพโปรไฟล์
        profile_face_encoding = await load_profile_image(request.profile_image)

        # คำนวณเอกลักษณ์ใบหน้าในภาพที่ส่งเข้ามา
        face_encodings = face.face_encodings(img, face_locations)

        # เปรียบเทียบใบหน้า
        for face_encoding in face_encodings:
            face_distances = face.face_distance([profile_face_encoding], face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # ตรวจสอบว่ามีความคล้ายกันที่ต่ำกว่าเกณฑ์
            if face_distances[best_match_index] < 0.6:
                return ImageResponse(match=True, distance=face_distances[best_match_index], message="พบใบหน้าที่ตรงกัน")
        
        # หากไม่พบใบหน้าที่ตรงกัน
        return ImageResponse(match=False, distance=None, message="ไม่พบใบหน้าที่ตรงกัน")

    except HTTPException as http_exc:
        raise http_exc
