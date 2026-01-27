import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

try:
    from answers import ETS_DATA
except ImportError:
    ETS_DATA = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# [ANSWERS_DB 생략 - 이전과 동일]

def super_enhance(img):
    """채도와 대비를 극대화하여 연필 자국을 컴싸처럼 만듦"""
    # 채도 강화
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 60) 
    enhanced = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    # 대비 강화
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.6, beta=-40)
    return enhanced

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(None)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        # 1. 이미지 선명화 (강사님 수동 보정 효과)
        image = super_enhance(image)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 가우시안 블러를 줄여 연필의 미세한 질감을 살림
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        target_regions = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]) 

        total_student_answers = []
        labels = ["A", "B", "C", "D"]

        for idx, c in enumerate(target_regions):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                approx = np.array(cv2.boxPoints(cv2.minAreaRect(c)), dtype="float32")
            else:
                approx = approx.reshape(4, 2).astype("float32")

            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=1); rect[0] = approx[np.argmin(s)]; rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=1); rect[1] = approx[np.argmin(diff)]; rect[3] = approx[np.argmax(diff)]

            dst_w, dst_h = 800, 600
            dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (dst_w, dst_h))
            
            # 3. 연필 마킹용 이진화 (BlockSize는 키우고 C는 줄임)
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
            
            # 4. 연필 자국 팽창
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            l_margin, t_margin = dst_w * 0.0842, dst_h * 0.1635  
            c_gap, r_gap = dst_w * 0.1988, dst_h * 0.0421     
            b_w = dst_w * 0.0342       

            for col in range(5):
                for row in range(20):
                    bx, by = l_margin + (col * c_gap), t_margin + (row * r_gap)
                    p_counts = []
                    for j in range(4):
                        cx, cy = int(bx + (j * b_w)), int(by)
                        mask = np.zeros((dst_h, dst_w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), 8, 255, -1) 
                        count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        p_counts.append(count)
                    
                    # 기준치를 10으로 낮춰 흐릿한 연필 포착
                    if max(p_counts) > 10: 
                        total_student_answers.append(labels[np.argmax(p_counts)])
                    else:
                        total_student_answers.append("?")

        # [정답 대조 로직 및 리턴 - 이전과 동일]
        # ...
