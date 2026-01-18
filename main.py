import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# 아임웹과의 통신을 허용하는 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [강사님 확인!] 테스트용 정답지 (나중에 실제 정답으로 수정하세요) ---
ANSWER_KEY = (["A", "B", "C", "D"] * 50)  # 200번까지 A,B,C,D 반복 (임시)

@app.get("/")
def home():
    return "TOEIC AI Server is Running!"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    # 2. 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 3. 테두리 찾기
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_regions = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 50000:
            target_regions.append(approx)
            if len(target_regions) == 2: break

    target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))

    if len(target_regions) == 0:
        return {"error": "답안지 구역(LC/RC)을 찾지 못했습니다."}

    total_student_answers = []
    labels = ["A", "B", "C", "D"]

    # 4. 각 구역(LC, RC) 분석
    for idx, region in enumerate(target_regions):
        section_name = "LC" if idx == 0 else "RC"
        
        # ROI 추출 및 정렬
        pts = region.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        
        dst_w, dst_h = 600, 800 
        dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h, w = warped.shape[:2]
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(warped_gray, 160, 255, cv2.THRESH_BINARY_INV)

        # 강사님의 황금 좌표 적용
        top_margin = h * 0.163
        row_spacing = h * 0.0422
        bubble_width = w * 0.034
        
        if section_name == "RC":
            left_margin = w * 0.080
            col_spacing = w * 0.196
        else:
            left_margin = w * 0.082
            col_spacing = w * 0.201

        # 100문항 순회
        for col in range(5):
            for row in range(20):
                base_x = left_margin + (col * col_spacing)
                base_y = top_margin + (row * row_spacing)
                
                pixel_counts = []
                for j in range(4):
                    cx, cy = int(base_x + (j * bubble_width)), int(base_y)
                    mask = np.zeros(warped_gray.shape, dtype="uint8")
                    cv2.circle(mask, (cx, cy), 6, 255, -1)
                    count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                    pixel_counts.append(count)

                if max(pixel_counts) > 25:
                    total_student_answers.append(labels[np.argmax(pixel_counts)])
                else:
                    total_student_answers.append("?")

    # 5. 채점 및 결과 반환
    score = 0
    results = []
    for i in range(len(total_student_answers)):
        std_ans = total_student_answers[i]
        is_correct = (std_ans == ANSWER_KEY[i])
        if is_correct: score += 1
        results.append({"no": i+1, "student": std_ans, "result": "O" if is_correct else "X"})

    return {"total_score": score, "results": results}
