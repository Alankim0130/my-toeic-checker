import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# 아임웹과의 통신 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [강사님 필독! 회차별 정답지 설정] ---
# 실제 정답에 맞춰 알파벳을 수정하세요.
ANSWERS_DB = {
    "vol16": ["A", "B", "C", "D"] * 50, # vol16 실제 정답 200개
    "vol17": ["B", "C", "D", "A"] * 50, # vol17 실제 정답 200개
}

@app.get("/")
def home():
    return "TOEIC AI Server is Running!"

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    vol: str = Form(...)  # 아임웹에서 보낸 회차 정보를 여기서 받습니다.
):
    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    # 2. 선택된 회차의 정답지 가져오기
    # 아임웹에서 보낸 vol 값이 ANSWERS_DB에 없으면 vol16을 기본값으로 사용
    ANSWER_KEY = ANSWERS_DB.get(vol, ANSWERS_DB["vol16"])

    # 3. 이미지 전처리 및 구역(LC/RC) 찾기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

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

    if len(target_regions) < 2:
        return {"error": "답안지 구역(LC/RC)을 모두 찾지 못했습니다. 사진을 다시 찍어주세요."}

    total_student_answers = []
    labels = ["A", "B", "C", "D"]

    # 4. 각 구역(LC, RC) 마킹 판독
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
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h, w = warped.shape[:2]
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(warped_gray, 160, 255, cv2.THRESH_BINARY_INV)

        # 좌표 설정
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
                    cv2.circle(mask, (cx, cy), 6, 255, -1)
                    pixel_counts.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))

                if max(pixel_counts) > 25:
                    total_student_answers.append(labels[np.argmax(pixel_counts)])
                else:
                    total_student_answers.append("?")

    # 5. 파트별 세분화 채점 로직
    parts_def = [
        ("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100),
        ("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)
    ]
    
    total_score = 0
    part_details = []
    for name, start, end in parts_def:
        p_score, p_items = 0, []
        for i in range(start-1, end):
            if i >= len(total_student_answers): break
            std = total_student_answers[i]
            corr = (std == ANSWER_KEY[i])
            if corr: 
                total_score += 1
                p_score += 1
            p_items.append({"no": i+1, "std": std, "res": "O" if corr else "X"})
        part_details.append({
            "name": name, 
            "score": p_score, 
            "total": end-start+1, 
            "items": p_items
        })

    return {"total_score": total_score, "part_details": part_details}
