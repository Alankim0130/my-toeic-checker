import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

# [확인] answers.py 파일 연동
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

ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    vol: str = Form(None)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
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
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
            
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

            # [85번 최종 정밀 보정]
            l_margin = dst_w * 0.0842  # 0.0845 -> 0.0842 보정
            t_margin = dst_h * 0.1635  
            c_gap = dst_w * 0.1985     
            r_gap = dst_h * 0.0421     
            b_w = dst_w * 0.0342       

            for col in range(5):
                for row in range(20):
                    bx = l_margin + (col * c_gap)
                    by = t_margin + (row * r_gap)
                    p_counts = []
                    for j in range(4):
                        cx, cy = int(bx + (j * b_w)), int(by)
                        mask = np.zeros((dst_h, dst_w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), 6, 255, -1) 
                        count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        p_counts.append(count)
                    
                    if max(p_counts) > 22:
                        total_student_answers.append(labels[np.argmax(p_counts)])
                    else:
                        total_student_answers.append("?")

        # --- 정답 대조 및 채점 ---
        ANSWER_KEY = ["-"] * 200
        clean_vol = vol.replace(".", "") if vol else "vol16"
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])

        lc_correct, rc_correct, part_details = 0, 0, []
        p_defs = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100),
                  ("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]

        for name, s, e in p_defs:
            p_score, p_items = 0, []
            for i in range(s-1, e):
                std = total_student_answers[i] if i < len(total_student_answers) else "?"
                ans = ANSWER_KEY[i]
                corr = (std == ans) if ans != "-" else False
                if corr:
                    p_score += 1
                    if i < 100: lc_correct += 1
                    else: rc_correct += 1
                p_items.append({"no": i+1, "std": std, "ans": ans, "res": "O" if corr else "X"})
            part_details.append({"name": name, "score": p_score, "total": e-s+1, "items": p_items})

        lc_conv = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        rc_conv = min(rc_correct * 5, 495)

        # 아임웹 자바스크립트가 기대하는 키값 매칭
        return {
            "lc_correct": lc_correct,   # 아임웹 화면 노출용
            "rc_correct": rc_correct,   # 아임웹 화면 노출용
            "lc_converted": lc_conv,
            "rc_converted": rc_conv,
            "total_converted": lc_conv + rc_conv,
            "part_details": part_details
        }

    except Exception as e:
        return {"error": str(e)}
