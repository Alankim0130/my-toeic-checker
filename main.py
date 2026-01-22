import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

# [확인] answers.py 파일에서 ETS_DATA를 불러옵니다.
from answers import ETS_DATA 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# [유지] 기존 vol16, vol17 정답 데이터 (이건 main.py에 그대로 둡니다)
ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    
    # 기존용 파라미터 (값이 없으면 None)
    vol: str = Form(None),       
    
    # ETS용 파라미터 (값이 없으면 None)
    textbook: str = Form(None),  
    lc_round: str = Form(None),  
    rc_round: str = Form(None),

    # 학생 정보 (ETS는 안 보내므로 None 처리 필수)
    name: str = Form(None),
    last4: str = Form(None)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        # 1. 정답지 초기화 (200개 모두 '-'로 채움)
        ANSWER_KEY = ["-"] * 200

        # --- [정답지 로드 로직: answers.py 연동] ---
        
        # Case A: ETS 기출문제 (textbook 값이 있는 경우)
        if textbook and textbook in ETS_DATA:
            # LC 정답 병합 (1~100번)
            if lc_round and lc_round != "none":
                # answers.py의 ETS_DATA에서 가져옴
                lc_list = ETS_DATA[textbook]["LC"].get(lc_round, [])
                limit = min(len(lc_list), 100)
                ANSWER_KEY[0:limit] = lc_list[0:limit]
            
            # RC 정답 병합 (101~200번)
            if rc_round and rc_round != "none":
                # answers.py의 ETS_DATA에서 가져옴
                rc_list = ETS_DATA[textbook]["RC"].get(rc_round, [])
                limit = min(len(rc_list), 100)
                # 101번(인덱스 100)부터 채움
                ANSWER_KEY[100:100+limit] = rc_list[0:limit]

        # Case B: 기존 교재 (vol 값이 있는 경우)
        elif vol:
            clean_vol = vol.replace(".", "")
            ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])
        
        # --- [OpenCV 판독 로직 (기존과 동일)] ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        
        target_regions = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
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

        # --- [가변 채점: 선택한 파트만 UI에 표시] ---
        lc_parts_def = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100)]
        rc_parts_def = [("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]
        
        target_parts = []

        # (1) LC 파트 포함 여부 결정
        # 기존 교재(vol)이거나, ETS LC회차가 선택된 경우
        if vol or (lc_round and lc_round != "none"):
            target_parts.extend(lc_parts_def)
            
        # (2) RC 파트 포함 여부 결정
        # 기존 교재(vol)이거나, ETS RC회차가 선택된 경우
        if vol or (rc_round and rc_round != "none"):
            target_parts.extend(rc_parts_def)

        lc_correct, rc_correct, part_details = 0, 0, []
        db_answer_length = len(ANSWER_KEY)

        # (3) 포함된 파트만 순회하며 채점
        for name, start, end in target_parts:
            p_score, p_items = 0, []
            for i in range(start-1, end):
                # 학생 답안 가져오기
                std = total_student_answers[i] if i < len(total_student_answers) else "?"
                
                # 정답 DB 대조
                if i < db_answer_length:
                    ans = ANSWER_KEY[i]
                    # 정답이 '-' (미선택 파트)라면 무조건 False
                    corr = (std == ans) if ans != "-" else False
                else:
                    ans = "-" 
                    corr = False
                
                if corr:
                    if i < 100: lc_correct += 1
                    else: rc_correct += 1
                    p_score += 1
                
                p_items.append({
                    "no": i+1, 
                    "std": std, 
                    "ans": ans, 
                    "res": "O" if corr else ("-" if ans == "-" else "X")
                })
            part_details.append({"name": name, "score": p_score, "total": end-start+1, "items": p_items})

        # --- [점수 환산 및 리턴] ---
        lc_converted = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        rc_converted = min(rc_correct * 5, 495)
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
