import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# 1. 아임웹과의 통신 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --- [정답지 설정] ---
ANSWERS_DB = {
    "vol16": [
        "C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C",
        "A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C",
        "B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A",
        "A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D",
        "B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A",
        "B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B",
        "C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D",
        "A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C",
        "B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D",
        "C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"
    ],
    "vol17": [
        "D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B",
        "B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D",
        "B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C",
        "A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D",
        "C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C",
        "C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A",
        "B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C",
        "A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B",
        "C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C",
        "B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"
    ],
}

@app.get("/")
def home():
    return {"status": "TOEIC AI Server is Running!"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    clean_vol = vol.replace(".", "")
    ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_regions = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 30000:
            target_regions.append(approx)
            if len(target_regions) == 2: break

    if len(target_regions) < 2:
        return {"error": "답안지 구역을 찾지 못했습니다. 밝은 곳에서 테두리가 잘 보이게 찍어주세요."}

    target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))

    total_student_answers = []
    labels = ["A", "B", "C", "D"]

    for idx, region in enumerate(target_regions):
        section_name = "LC" if idx == 0 else "RC"
        pts = region.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        
        # 1. 원근 변환 (강사님의 최적 비율 600x800)
        dst_w, dst_h = 600, 800 
        dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        
        # 2. 지능형 회전 (가로로 누운 사진일 때만 왼쪽으로 90도 회전)
        if warped.shape
