import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

@app.get("/")
def home():
    return {"status": "TOEIC AI Server is Running!"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        ANSWER_KEY = ANSWERS_DB.get(vol.replace(".", ""), ANSWERS_DB["vol16"])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        target_regions = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 테두리 인식 면적 기준을 살짝 낮춰 인식률 확보
            if len(approx) == 4 and cv2.contourArea(c) > 15000:
                target_regions.append(approx)
                if len(target_regions) == 2: break

        if len(target_regions) < 2:
            return {"error": "LC/RC 구역 인식 실패. 테두리가 잘 보이게 찍어주세요."}

        # 왼쪽(LC), 오른쪽(RC) 정렬
        target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))

        total_student_answers = []
        labels = ["A", "B", "C", "D"]

        for idx, region in enumerate(target_regions):
            section_name = "LC" if idx == 0 else "RC"
            
            pts = region.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
            
            # [핵심] 각각을 600x800으로 변환
            dst_w, dst_h = 600, 800 
            dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
            
            # --- [복구] 강제 회전 제거 (600x800 세로 상태 그대로 판독) ---
            h, w = warped.shape[:2]
            w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(w_gray, 150, 255, cv2.THRESH_BINARY_INV)

            # --- 강사님 황금 좌표 (600x800 기준) ---
            top_margin = h * 0.163
            row_spacing = h * 0.0422
            
            if section_name == "RC":
                left_margin = w * 0.080 
                col_spacing = w * 0.196 
                bubble_w = w * 0.033
            else:
                left_margin = w * 0.082
                col_spacing = w * 0.201
                bubble_w = w * 0.034

            for col in range(5):
                for row in range(20):
                    base_x = left_margin + (col * col_spacing)
                    base_y = top_margin + (row * row_spacing)
                    
                    pixel_counts = []
                    for j in range(4):
                        cx, cy = int(base_x + (j * bubble_w)), int(base_y)
                        
                        # 좌표 안전성 체크
                        if cx < 0 or cx >= w or cy < 0 or cy >= h:
                            pixel_counts.append(0)
                            continue

                        mask = np.zeros((h, w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), 6, 255, -1)
                        pixel_counts.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))

                    if max(pixel_counts) > 25:
                        total_student_answers.append(labels[np.argmax(pixel_counts)])
                    else:
                        total_student_answers.append("?")

        # 결과 리턴 로직
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
            "lc_correct": lc_correct, "rc_correct": rc_correct,
            "lc_converted": lc_correct * 5, "rc_converted": rc_correct * 5,
            "total_converted": (lc_correct + rc_correct) * 5, "part_details": part_details
        }
    except Exception as e:
        return {"error": f"서버 오류: {str(e)}"}
