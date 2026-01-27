import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io

# [확인] answers.py 파일에서 ETS_DATA를 불러옵니다.
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
    vol: str = Form(None),       
    textbook: str = Form(None),  
    lc_round: str = Form(None),  
    rc_round: str = Form(None),
    name: str = Form(None),
    last4: str = Form(None)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return {"error": "이미지 읽기 실패"}

        # --- [전략 1] 강제 왼쪽 90도 회전 및 대비 보정 (CLAHE) ---
        # 세로로 업로드된 사진을 가로형 정방향으로 전환
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 조명 불균형 해소를 위한 CLAHE 적용
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

        # --- [전략 2] 스마트 박스 검출 (가장 큰 사각형 2개) ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 면적 순으로 정렬하여 가장 큰 덩어리 2개만 선별
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        
        # 왼쪽(LC), 오른쪽(RC) 순서로 정렬
        target_regions = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

        total_student_answers = []
        labels = ["A", "B", "C", "D"]

        for idx, c in enumerate(target_regions):
            # --- [전략 3] 원근 보정 (Perspective Transform) ---
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # 4개 꼭짓점이 안 잡힐 경우 최소 사각형으로 대체
            if len(approx) != 4:
                rect_box = cv2.minAreaRect(c)
                approx = np.array(cv2.boxPoints(rect_box), dtype="float32")
            else:
                approx = approx.reshape(4, 2).astype("float32")

            # 좌표 정렬 (좌상, 우상, 우하, 좌하)
            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=1); rect[0] = approx[np.argmin(s)]; rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=1); rect[1] = approx[np.argmin(diff)]; rect[3] = approx[np.argmax(diff)]

            # 강사님 지침: 모든 박스를 가로형 800x600으로 리사이즈
            dst_w, dst_h = 800, 600
            dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h))

            # --- [전략 4] 음영 제거 및 중심부 샘플링 판독 ---
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # 그림자 지우는 Adaptive Threshold
            thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)

            # 800x600 기준 정밀 좌표 (비율 기반)
            left_margin, top_margin = dst_w * 0.082, dst_h * 0.163
            col_spacing, row_spacing = dst_w * 0.201, dst_h * 0.0422
            bubble_width = dst_w * 0.034

            for col in range(5):
                for row in range(20):
                    base_x = left_margin + (col * col_spacing)
                    base_y = top_margin + (row * row_spacing)
                    pixel_ratios = []
                    
                    for j in range(4):
                        cx, cy = int(base_x + (j * bubble_width)), int(base_y)
                        
                        # [전략 5] 중심부 샘플링 (반지름을 작게 잡아 글자 간섭 최소화)
                        mask = np.zeros((dst_h, dst_w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), 5, 255, -1) # 반지름 5로 타이트하게 설정
                        
                        # 검은색 픽셀 농도 측정
                        pixel_count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        # 원 면적 대비 점유율 계산 (글자는 70%를 넘기 힘듬)
                        pixel_ratios.append(pixel_count)

                    # 임계값: 글자 오인을 막기 위해 픽셀 점유율이 일정 수준 이상인 것만 인정
                    if max(pixel_ratios) > 25: 
                        total_student_answers.append(labels[np.argmax(pixel_ratios)])
                    else:
                        total_student_answers.append("?")

        # --- [정답 대조 및 환산 로직 (기존 유지)] ---
        ANSWER_KEY = ["-"] * 200
        if textbook and textbook in ETS_DATA:
            if lc_round and lc_round != "none":
                lc_list = ETS_DATA[textbook]["LC"].get(lc_round, [])
                ANSWER_KEY[0:min(len(lc_list), 100)] = lc_list[0:100]
            if rc_round and rc_round != "none":
                rc_list = ETS_DATA[textbook]["RC"].get(rc_round, [])
                ANSWER_KEY[100:100+min(len(rc_list), 100)] = rc_list[0:100]
        elif vol:
            ANSWER_KEY = ANSWERS_DB.get(vol.replace(".", ""), ANSWERS_DB["vol16"])

        lc_parts_def = [("Part 1", 1, 6), ("Part 2", 7, 31), ("Part 3", 32, 70), ("Part 4", 71, 100)]
        rc_parts_def = [("Part 5", 101, 130), ("Part 6", 131, 146), ("Part 7", 147, 200)]
        
        target_parts = []
        if vol or (lc_round and lc_round != "none"): target_parts.extend(lc_parts_def)
        if vol or (rc_round and rc_round != "none"): target_parts.extend(rc_parts_def)

        lc_correct, rc_correct, part_details = 0, 0, []
        for name, start, end in target_parts:
            p_score, p_items = 0, []
            for i in range(start-1, end):
                std = total_student_answers[i] if i < len(total_student_answers) else "?"
                ans = ANSWER_KEY[i]
                corr = (std == ans) if ans != "-" else False
                if corr:
                    if i < 100: lc_correct += 1
                    else: rc_correct += 1
                    p_score += 1
                p_items.append({"no": i+1, "std": std, "ans": ans, "res": "O" if corr else ("-" if ans == "-" else "X")})
            part_details.append({"name": name, "score": p_score, "total": end-start+1, "items": p_items})

        lc_converted = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        rc_converted = min(rc_correct * 5, 495)
        
        return {
            "lc_correct": lc_correct, "rc_correct": rc_correct,
            "lc_converted": lc_converted, "rc_converted": rc_converted,
            "total_converted": lc_converted + rc_converted,
            "part_details": part_details
        }
    except Exception as e:
        return {"error": f"분석 오류: {str(e)}"}
