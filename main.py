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

def smart_lighting_balance(img):
    """강사님 요청대로 어두운 곳만 골라 밝히는 지능형 전처리"""
    # 1. 흑백 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE 적용 (국소 대비 강화) - 어두운 RC 영역의 연필 자국을 도드라지게 함
    # clipLimit를 높여 어두운 곳을 더 강하게 밝힘
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
    gray = clahe.apply(gray)
    
    # 3. 전체 밝기 정규화 - 너무 밝은 곳은 억제하고 어두운 곳은 기준치까지 상승
    # 이미지의 1%~99% 픽셀 범위를 0~255로 쫙 펼칩니다.
    min_val, max_val = np.percentile(gray, (1, 99))
    gray = np.clip(gray, min_val, max_val)
    gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return gray

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(None)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        # 정방향 회전
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # [핵심] 지능형 조명 밸런싱 적용
        gray_processed = smart_lighting_balance(image)

        # 박스 찾기 (Canny 에지 검출)
        edged = cv2.Canny(cv2.GaussianBlur(gray_processed, (5, 5), 0), 30, 150)
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
            warped = cv2.warpPerspective(gray_processed, M, (dst_w, dst_h))
            
            # 연필 마킹 인식을 위해 감도 대폭 개방 (BlockSize=51, C=2)
            # C값이 낮을수록 흐릿한 연필 마킹을 검은색으로 더 잘 잡아냅니다.
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
            
            # 연필 가루를 뭉쳐서 확실하게 만들기
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
                        cv2.circle(mask, (cx, cy), 9, 255, -1) # 반지름 9로 확장
                        count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        p_counts.append(count)
                    
                    # 조명을 맞췄으므로 기준치를 다시 20 정도로 세팅
                    if max(p_counts) > 20: 
                        total_student_answers.append(labels[np.argmax(p_counts)])
                    else:
                        total_student_answers.append("?")

        # --- 채점 및 리턴 로직 (이전과 동일) ---
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

        return {
            "lc_correct": lc_correct, "rc_correct": rc_correct,
            "lc_converted": min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0,
            "rc_converted": min(rc_correct * 5, 495),
            "total_converted": (min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0) + min(rc_correct * 5, 495),
            "part_details": part_details
        }
    except Exception as e:
        return {"error": str(e)}
