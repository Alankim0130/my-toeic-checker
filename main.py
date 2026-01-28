import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

# [필독] ANSWERS_DB 누락 없이 포함
ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def advanced_lighting_fix(img):
    """강사님 요청: 어두운 곳은 밝게, 밝은 곳은 유지"""
    # 1. 흑백 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 적응형 조명 보정 (CLAHE) - 국소적인 명암 차이를 극대화
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. 섀도우 복원 (Shadow Recovery)
    # 이미지의 어두운 부분을 찾아내어 그 부분에만 가중치를 줌
    thresh_val = np.mean(gray)
    mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.GaussianBlur(mask, (31, 31), 0) / 255.0
    
    # 어두운 영역은 1.4배 밝게, 밝은 영역은 그대로 유지
    result = gray.astype(float) + (mask * gray.astype(float) * 0.4)
    return np.clip(result, 0, 255).astype(np.uint8)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(None)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # [핵심] 조명 평준화 전처리 실행
        gray_balanced = advanced_lighting_fix(image)
        gray_balanced = cv2.rotate(gray_balanced, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 박스 검출 (조명이 평준화되어 에지가 더 선명함)
        edged = cv2.Canny(cv2.GaussianBlur(gray_balanced, (5, 5), 0), 30, 150)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        target_regions = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

        total_student_answers = []
        labels = ["A", "B", "C", "D"]

        for idx, c in enumerate(target_regions):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4: approx = np.array(cv2.boxPoints(cv2.minAreaRect(c)), dtype="float32")
            else: approx = approx.reshape(4, 2).astype("float32")

            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=1); rect[0] = approx[np.argmin(s)]; rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=1); rect[1] = approx[np.argmin(diff)]; rect[3] = approx[np.argmax(diff)]

            dst_w, dst_h = 800, 600
            M = cv2.getPerspectiveTransform(rect, np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype="float32"))
            warped = cv2.warpPerspective(gray_balanced, M, (dst_w, dst_h))
            
            # 조명이 평준화되었으므로 고감도 이진화 적용
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
            
            # 마킹 판독 (좌표는 이전과 동일)
            l_margin, t_margin = dst_w * 0.0842, dst_h * 0.1635  
            c_gap, r_gap = dst_w * 0.1988, dst_h * 0.0421     
            b_w = dst_w * 0.0342       

            for col in range(5):
                for row in range(20):
                    bx, by = l_margin + (col * c_gap), t_margin + (row * r_gap)
                    p_counts = [cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=cv2.circle(np.zeros((dst_h, dst_w), dtype="uint8"), (int(bx+(j*b_w)), int(by)), 9, 255, -1))) for j in range(4)]
                    
                    if max(p_counts) > 15: total_student_answers.append(labels[np.argmax(p_counts)])
                    else: total_student_answers.append("?")

        # 채점 로직
        clean_vol = vol.replace(".", "") if vol else "vol16"
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])
        lc_correct = sum(1 for i in range(100) if total_student_answers[i] == ANSWER_KEY[i])
        rc_correct = sum(1 for i in range(100, 200) if total_student_answers[i] == ANSWER_KEY[i])

        return {
            "lc_correct": lc_correct, "rc_correct": rc_correct,
            "lc_converted": min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0,
            "rc_converted": min(rc_correct * 5, 495),
            "total_converted": (min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0) + min(rc_correct * 5, 495)
        }
    except Exception as e:
        return {"error": str(e)}
