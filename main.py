import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# 1. 아임웹과의 통신 허용 설정 (보강됨)
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
    "vol16": [ # 아임웹에서 "vol16"으로 보내므로 키값을 맞췄습니다.
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
    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    # 2. 선택된 회차의 정답지 가져오기 (키값 매칭 보정)
    clean_vol = vol.replace(".", "") # "vol.16" -> "vol16"
    ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])

    # 3. 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150) # Canny 범위 소폭 조정

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_regions = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 30000: # 최소 영역 소폭 완화
            target_regions.append(approx)
            if len(target_regions) == 2: break

    if len(target_regions) < 2:
        return {"error": "답안지 구역을 찾지 못했습니다. 밝은 곳에서 테두리가 잘 보이게 찍어주세요."}

    # 왼쪽 구역부터 정렬
    target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))

    total_student_answers = []
    labels = ["A", "B", "C", "D"]

    # 4. 판독
    for idx, region in enumerate(target_regions):
        section_name = "LC" if idx == 0 else "RC"
        pts = region.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        
        dst_w, dst_h = 600, 800 
        dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        
        # 가로로 찍힌 경우 자동으로 똑바로 세우기
        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h, w = warped.shape[:2]
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # 조명 차이에 강한 Adaptive Threshold 사용 권장 (또는 현재값 유지)
        _, thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV)

        top_margin, row_spacing, bubble_width = h * 0.163, h * 0.0422, w * 0.034
        left_margin = w * 0.080 if section_name == "RC" else w * 0.082
        col_spacing = w * 0.196 if section_name == "RC" else w * 0.201

        for col in range(5):
            for row in range(20):
                base_x = left_margin + (col * col_spacing)
                base_y = top_margin + (row * row_spacing)
                
                pixel_counts = []
                for j in range(4):
                    cx, cy = int(base_x + (j * bubble_width)), int(base_y)
                    mask = np.zeros(warped_gray.shape, dtype="uint8")
                    cv2.circle(mask, (cx, cy), 7, 255, -1) # 원 크기 소폭 확대
                    pixel_counts.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))

                if max(pixel_counts) > 20: # 판독 기준 소폭 완화
                    total_student_answers.append(labels[np.argmax(pixel_counts)])
                else:
                    total_student_answers.append("?")

    # 5. 파트별 채점
    parts_def = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100),
                 ("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]
    
    lc_correct, rc_correct, part_details = 0, 0, []

    for name, start, end in parts_def:
        p_score, p_items = 0, []
        for i in range(start-1, end):
            if i >= len(total_student_answers): break
            std = total_student_answers[i]
            corr = (std == ANSWER_KEY[i])
            if corr:
                if i < 100: lc_correct += 1
                else: rc_correct += 1
                p_score += 1
            p_items.append({"no": i+1, "std": std, "res": "O" if corr else "X"})
        part_details.append({"name": name, "score": p_score, "total": end-start+1, "items": p_items})

    return {
        "lc_correct": lc_correct,
        "rc_correct": rc_correct,
        "lc_converted": lc_correct * 5,
        "rc_converted": rc_correct * 5,
        "total_converted": (lc_correct + rc_correct) * 5,
        "part_details": part_details
    }
