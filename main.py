import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

# [필독] 정답 데이터베이스 - 이제 절대 누락되지 않습니다.
ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def smart_shadow_recovery(img):
    """지능형 섀도우 복원: 강사님이 직접 채도/밝기를 만진 것과 같은 효과"""
    # 흑백 변환 후 국소 대비 강화 (어두운 곳 위주로 대비 상승)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(12,12)) # RC 영역 저격 수치
    gray = clahe.apply(gray)
    
    # 조명 맵 생성: 평균보다 어두운 영역을 찾아 1.5배 밝게 보정
    blur_gray = cv2.GaussianBlur(gray, (61, 61), 0)
    mean_val = np.mean(blur_gray)
    shadow_mask = cv2.threshold(blur_gray, mean_val, 255, cv2.THRESH_BINARY_INV)[1].astype(float) / 255.0
    
    # 결과 합성: 어두운 부분만 선택적 보정
    result = gray.astype(float) + (shadow_mask * gray.astype(float) * 0.5)
    return np.clip(result, 0, 255).astype(np.uint8)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(None)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. 지능형 전처리 및 정방향 회전
        gray_balanced = smart_shadow_recovery(image)
        gray_balanced = cv2.rotate(gray_balanced, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. 박스 및 그리드 추출
        edged = cv2.Canny(cv2.GaussianBlur(gray_balanced, (5, 5), 0), 30, 150)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        target_regions = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

        student_all_ans = []
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
            
            # 3. 연필 마킹 전용 고감도 판독
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 4)

            # 검증된 정밀 좌표 수치 (85번 해결값 유지)
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
                        cv2.circle(mask, (cx, cy), 10, 255, -1) # 원 지름을 10으로 키워 포착력 상승
                        count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        p_counts.append(count)
                    
                    if max(p_counts) > 15: student_all_ans.append(labels[np.argmax(p_counts)])
                    else: student_all_ans.append("?")

        # 4. 아임웹 상세 내역(part_details) 생성 - forEach 에러 방지
        clean_vol = vol.replace(".", "") if vol else "vol16"
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])
        
        lc_correct, rc_correct, part_details = 0, 0, []
        p_defs = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100),
                  ("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]

        for name, s, e in p_defs:
            p_score, p_items = 0, []
            for i in range(s-1, e):
                std = student_all_ans[i] if i < len(student_all_ans) else "?"
                ans = ANSWER_KEY[i]
                is_corr = (std == ans)
                if is_corr:
                    p_score += 1
                    if i < 100: lc_correct += 1
                    else: rc_correct += 1
                p_items.append({"no": i+1, "std": std, "ans": ans, "res": "O" if is_corr else "X"})
            part_details.append({"name": name, "score": p_score, "total": e-s+1, "items": p_items})

        return {
            "lc_correct": lc_correct, "rc_correct": rc_correct,
            "lc_converted": min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0,
            "rc_converted": min(rc_correct * 5, 495),
            "total_converted": (min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0) + min(rc_correct * 5, 495),
            "part_details": part_details
        }
    except Exception as e:
        return {"error": str(e)}
