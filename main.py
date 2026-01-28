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

def adaptive_lighting_balance(img):
    """지능형 조명 밸런싱: 어두운 곳만 골라서 밝게 만듦"""
    # 1. 감마 보정을 통해 이미지의 동적 범위 확보
    # 전체 이미지의 평균 밝기를 계산하여 감마값 자동 결정
    mid = 0.5
    mean = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
    gamma = np.log(mid) / np.log(mean)
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 2. CLAHE (적응형 히스토그램 평활화)를 통해 국소적 대비 강화
    # 이미 밝은 곳은 놔두고, 어두운 곳의 대비만 끌어올림
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12,12)) # 타일 크기를 키워 자연스럽게 보정
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    
    return img

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), vol: str = Form(None)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        # [핵심] 학생별 촬영 환경 맞춤형 보정 적용
        image = adaptive_lighting_balance(image)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
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
            
            # 연필 마킹 포착을 위해 가우시안 임계값 미세 조정
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)

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
                    
                    # 수동 보정에서 효과가 좋았던 기준값 25 전후로 세팅
                    if max(p_counts) > 22: 
                        total_student_answers.append(labels[np.argmax(p_counts)])
                    else:
                        total_student_answers.append("?")

        # --- 정답 대조 및 리턴 ---
        lc_correct, rc_correct, part_details = 0, 0, []
        clean_vol = vol.replace(".", "") if vol else "vol16"
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])
        
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
