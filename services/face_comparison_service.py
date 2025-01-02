import face_recognition as face
import numpy as np
from utils import decode_base64_image
from fastapi import HTTPException, status
from models import ImageRequest
import os 
async def compare_face_service(request : ImageRequest):
    try:
        # แปลง base64 ไปเป็นภาพ
        img = decode_base64_image(request.check_in_image)

        # ตรวจสอบใบหน้าในภาพ
        face_locations = face.face_locations(img, model="hog")
        if len(face_locations) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่พบใบหน้าในภาพที่ส่งมา")

        # ตรวจสอบว่า profile_image เป็น path ที่ถูกต้อง
        if not request.profile_image or not os.path.exists(request.profile_image):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบไฟล์ภาพในเซิร์ฟเวอร์")

        # ใบหน้าที่ต้องการเปรียบเทียบ
        try:
            profile_image = face.load_image_file(request.profile_image)
            profile_face_encoding = face.face_encodings(profile_image)
            if len(profile_face_encoding) == 0:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่พบใบหน้าจากภาพที่เก็บไว้")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการโหลดภาพจากเซิร์ฟเวอร์")

        profile_face_encoding = profile_face_encoding[0]

        # ค้นหาตำแหน่งใบหน้าและเอกลักษณ์ใบหน้าในภาพที่ส่งเข้ามา
        face_encodings = face.face_encodings(img, face_locations)

        # ตรวจสอบใบหน้าที่พบ
        for face_encoding in face_encodings:
            face_distances = face.face_distance([profile_face_encoding], face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:  # ใช้ค่าความคล้ายที่ต่ำกว่า 0.6
                return {"result": True, "distance": face_distances[best_match_index], "message": "พบใบหน้าที่ตรงกัน"}

        return {"result": False, "distance": face_distances[best_match_index], "message": "ไม่พบใบหน้าที่ตรงกัน"}
    except HTTPException as http_exc:
        raise http_exc