from io import BytesIO
from fastapi import FastAPI, File, Form,Request,HTTPException, UploadFile ,status
import face_recognition as face
import numpy as np
import os
from models.check_in_model import ImageResponse
app = FastAPI()


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


@app.post("/compare-face/")
async def compare_face_endpoint(file: UploadFile = File(...), profile: str = Form(...)):
    try:
        img = await file.read()
        result = await compare_face_service(profile, img)
        return result
    except HTTPException as e:
        raise e
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

async def compare_face_service(profile: str, img_data: bytes):
    try:
        img_file = BytesIO(img_data)
        img = face.load_image_file(img_file)
        # ตรวจสอบการมีใบหน้าในภาพที่ส่งมา
        face_locations = face.face_locations(img, model="hog")
        if len(face_locations) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่พบใบหน้าในภาพที่ส่งมา")
        # โหลดข้อมูลใบหน้าจากภาพโปรไฟล์
        profile_face_encoding = await load_profile_image(profile)

        # คำนวณเอกลักษณ์ใบหน้าในภาพที่ส่งเข้ามา
        face_encodings = face.face_encodings(img, face_locations)

        # เปรียบเทียบใบหน้า
        for face_encoding in face_encodings:
            face_distances = face.face_distance([profile_face_encoding], face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # ตรวจสอบว่ามีความคล้ายกันที่ต่ำกว่าเกณฑ์
            if face_distances[best_match_index] < 0.6:
                return ImageResponse(match=True,persent=(1-face_distances[best_match_index])*100, message="พบใบหน้าที่ตรงกัน")
        
        # หากไม่พบใบหน้าที่ตรงกัน
        return ImageResponse(match=False, persent=0, message="ไม่พบใบหน้าที่ตรงกัน")

    except HTTPException as http_exc:
        raise http_exc
