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
        edged = cv2.Canny(blurred, 75, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        
        target_regions = []
        for c in cnts:
            # --- [ValueError 해결 핵심: 사각형 근사화] ---
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # 만약 점이 4개가 아니면 최소 사각형(박스)으로 강제 변환
            if len(approx) != 4:
                rect_box = cv2.minAreaRect(c)
                approx = cv2.boxPoints(rect_box)
                approx = np.array(approx, dtype="int")
            
            target_regions.append(approx)

        target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))
        total_student_answers = []
        labels = ["A", "B", "C", "D"]

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

            # 강사님 황금 수치 (RC 181~187 보정값 적용)
            if section_name == "RC":
                left_margin, current_col_spacing, bubble_width = w*0.083, w*0.198, w*0.034
            else:
                left_margin, current_col_spacing, bubble_width = w*0.082, w*0.201, w*0.034
            top_margin, row_spacing = h*0.163, h*0.0422

            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            for col in range(5):
                for row in range(20):
                    base_x = left_margin + (col * current_col_spacing)
                    base_y = top_margin + (row * row_spacing)
                    
                    pixel_counts = []
                    for j in range(4):
                        cx, cy = int(base_x + (j * bubble_width)), int(base_y)
                        mask = np.zeros((h, w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), 7, 255, -1)
                        pixel_count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        pixel_counts.append(pixel_count)

                    if max(pixel_counts) > 20:
                        total_student_answers.append(labels[np.argmax(pixel_counts)])
                    else:
                        total_student_answers.append("?")

        # --- [수정된 결과 처리 섹션] ---
        parts_def = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100),
                     ("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]
        lc_correct, rc_correct, part_details = 0, 0, []
        
        for name, start, end in parts_def:
            p_score, p_items = 0, []
            for i in range(start-1, end):
                if i >= len(total_student_answers): break
                
                std = total_student_answers[i]    # 학생이 쓴 답
                ans = ANSWER_KEY[i]              # 실제 정답
                corr = (std == ans)
                
                if corr:
                    if i < 100: lc_correct += 1
                    else: rc_correct += 1
                    p_score += 1
                
                # 결과 리스트에 정답(ans)을 추가하여 전달합니다.
                p_items.append({
                    "no": i+1, 
                    "std": std, 
                    "ans": ans,                  # 정답 데이터 추가
                    "res": "O" if corr else "X"
                })
            part_details.append({"name": name, "score": p_score, "total": end-start+1, "items": p_items})

        # --- [수정된 점수 환산 로직 적용] ---
        
        # 1. LC 환산: (개수 * 5) + 10점 가산, 최대 495점 제한
        # 단, 0개 맞았을 때는 0점으로 표시 (기본 점수 5점이 필요하면 0 대신 5 설정 가능)
        lc_converted = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        
        # 2. RC 환산: (개수 * 5), 최대 495점 제한
        rc_converted = min(rc_correct * 5, 495)
        
        # 3. 총합 계산
        total_converted = lc_converted + rc_converted

        return {
            "lc_correct": lc_correct, 
            "rc_correct": rc_correct, 
            "lc_converted": lc_converted, 
            "rc_converted": rc_converted, 
            "total_converted": total_converted, 
            "part_details": part_details
        }
    except Exception as e:
        return {"error": f"분석 오류: {str(e)}"}


