import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# 스마트폰 촬영 고려 핵심 유틸
# =========================

def order_points(pts: np.ndarray) -> np.ndarray:
    """4점 정렬: tl, tr, br, bl"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def preprocess_for_detection(bgr: np.ndarray) -> np.ndarray:
    """
    OMR 영역(큰 박스) 찾기용 전처리:
    - 그레이
    - 조명 평탄화(배경 추정 후 나눗셈)
    - CLAHE(너무 과하지 않게)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 조명 평탄화 (스마트폰 그림자/부분 어두움 대응)
    k = 61  # 31~101 홀수. 너무 작으면 그림자 제거 약함, 너무 크면 경계 흐림
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)

    # 국소 대비 강화 (과하면 종이결/노이즈 폭증 -> clipLimit 낮게)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    return norm

def find_omr_regions(gray_norm: np.ndarray) -> list:
    """
    OMR의 큰 사각 영역(LC/RC 영역) 2개 찾기.
    스마트폰 환경에서 컨투어가 많이 나오므로:
    - edge -> contour
    - 면적/사각형성/종횡비 기반 필터
    """
    # edge
    edged = cv2.Canny(gray_norm, 40, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray_norm.shape[:2]
    img_area = h * w

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.03:   # 너무 작은 건 제거 (스마트폰 노이즈/텍스트)
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 사각형에 가까운 후보만
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            # minAreaRect로 4점 강제
            pts = np.array(cv2.boxPoints(cv2.minAreaRect(c)), dtype="float32")

        rect = order_points(pts)
        # 가로/세로 비율 필터 (OMR 큰 박스는 대략 직사각형)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        ww = max(widthA, widthB)
        hh = max(heightA, heightB)
        if hh <= 1 or ww <= 1:
            continue
        ar = ww / hh

        # 종횡비가 너무 이상하면 제외 (촬영된 A4 기준 대충 1.1~2.5 사이가 흔함)
        if ar < 1.0 or ar > 3.2:
            continue

        candidates.append((area, rect))

    # 면적 큰 순으로 정렬 후 상위 몇 개
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 최종 2개만 선택하되, 서로 겹치거나 너무 비슷하면 다음 후보로 보정
    regions = []
    for _, rect in candidates:
        if len(regions) == 0:
            regions.append(rect)
        else:
            # 중심점 거리로 중복 후보 제거
            c1 = rect.mean(axis=0)
            c0 = regions[0].mean(axis=0)
            if np.linalg.norm(c1 - c0) < min(w, h) * 0.15:
                continue
            regions.append(rect)

        if len(regions) == 2:
            break

    # 왼쪽->오른쪽 정렬 (LC/RC가 보통 좌/우)
    regions.sort(key=lambda r: r[:, 0].mean())
    return regions

def warp_region(gray_norm: np.ndarray, bgr: np.ndarray, rect: np.ndarray, dst_w=800, dst_h=600):
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_gray = cv2.warpPerspective(gray_norm, M, (dst_w, dst_h))
    warped_bgr  = cv2.warpPerspective(bgr,      M, (dst_w, dst_h))
    return warped_gray, warped_bgr

def make_thresh_for_marks(warped_gray: np.ndarray) -> np.ndarray:
    """
    마킹 검출용 이진화:
    - adaptive threshold (스마트폰 조명 변화 대응)
    - open(잡티 제거) -> dilate(연필 자국 강화)
    """
    h, w = warped_gray.shape[:2]

    # 워핑 크기에 따라 block size를 유연하게 (홀수)
    # 너무 작으면 조명 영향 커짐, 너무 크면 마킹이 얇게 잡힐 수 있음
    block = int(max(51, (min(w, h) // 10) | 1))  # 대략 60~80대가 자주 나옴
    if block % 2 == 0:
        block += 1
    C = 8

    th = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, C
    )

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
    return th

def pick_choice_from_counts(counts, min_pixels=22, ratio=1.25):
    """
    스마트폰 환경 대응 마킹 판정:
    - 절대 픽셀수 기준 + 2등 대비 비율 기준을 동시에 만족해야 선택
    """
    best = int(np.max(counts))
    idx = int(np.argmax(counts))
    sorted_counts = sorted([int(x) for x in counts])
    second = sorted_counts[-2] if len(sorted_counts) >= 2 else 0

    if best >= min_pixels and (best / (second + 1)) >= ratio:
        return idx
    return None

# =========================
# API
# =========================

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    vol: str = Form(None),
    debug: int = Form(0)  # 1이면 서버에 디버그 이미지 저장
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다. 다른 사진으로 다시 시도해주세요."}

        # 0) (스마트폰) 방향은 다양하므로, 여기서는 강제 회전하지 않고
        #    먼저 영역을 찾아보고 실패하면 90도씩 돌려서 재시도합니다.
        rotations = [
            ("0", image),
            ("90CCW", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ("180", cv2.rotate(image, cv2.ROTATE_180)),
            ("90CW", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ]

        regions = None
        chosen_bgr = None
        chosen_gray = None
        chosen_tag = None

        for tag, img_try in rotations:
            gray_norm = preprocess_for_detection(img_try)
            regs = find_omr_regions(gray_norm)
            if len(regs) == 2:
                regions = regs
                chosen_bgr = img_try
                chosen_gray = gray_norm
                chosen_tag = tag
                break

        if regions is None:
            return {
                "error": "OMR 영역(LC/RC)을 찾지 못했습니다. "
                         "가능하면 답안지를 화면에 꽉 차게, 밝은 곳에서, 흔들림 없이 촬영해주세요."
            }

        labels = ["A", "B", "C", "D"]
        student_all_ans = []
        debug_views = []

        # 1) 두 영역(보통 좌=LC, 우=RC) 각각 워핑 후 5x20=100문항씩 읽기
        for idx, rect in enumerate(regions):
            warped_gray, warped_bgr = warp_region(chosen_gray, chosen_bgr, rect, dst_w=800, dst_h=600)
            th = make_thresh_for_marks(warped_gray)

            # 디버그 시각화
            vis = warped_bgr.copy()

            # (기존 너의 보정값 유지하되, 스마트폰 변동을 고려해 약간 더 안정적인 범위로 유지)
            dst_w, dst_h = 800, 600
            l_margin = dst_w * 0.0842
            t_margin = dst_h * 0.1635
            c_gap    = dst_w * 0.1988
            r_gap    = dst_h * 0.0421
            b_w      = dst_w * 0.0342

            # 마스크 반지름: 워핑 크기에 비례 (너무 작으면 연필 못 잡음)
            r = 10

            for col in range(5):
                for row in range(20):
                    bx = l_margin + (col * c_gap)
                    by = t_margin + (row * r_gap)

                    counts = []
                    centers = []
                    for j in range(4):
                        cx = int(bx + (j * b_w))
                        cy = int(by)
                        centers.append((cx, cy))

                        mask = np.zeros((dst_h, dst_w), dtype="uint8")
                        cv2.circle(mask, (cx, cy), r, 255, -1)
                        cnt = cv2.countNonZero(cv2.bitwise_and(th, th, mask=mask))
                        counts.append(cnt)

                        if debug:
                            # 후보 점 표시(빨간 점)
                            cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

                    # 스마트폰 대응 판정(절대+비율)
                    choice = pick_choice_from_counts(counts, min_pixels=22, ratio=1.25)

                    if choice is None:
                        student_all_ans.append("?")
                    else:
                        student_all_ans.append(labels[choice])
                        if debug:
                            # 선택된 곳 표시(초록 원)
                            sx, sy = centers[choice]
                            cv2.circle(vis, (sx, sy), 8, (0, 255, 0), 2)

            if debug:
                debug_views.append(vis)
                # threshold도 저장해두면 진단에 매우 유용
                cv2.imwrite(f"dbg_thresh_region_{idx}.jpg", th)

        # 디버그 결과 합쳐서 저장
        if debug and len(debug_views) == 2:
            cv2.imwrite("debug_result.jpg", np.hstack(debug_views))

        # 2) 채점
        clean_vol = (vol.replace(".", "") if vol else "vol16").strip()
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])

        if len(student_all_ans) < 200:
            # 2영역이 정상인데도 문항수가 부족하면 좌표가 틀어진 가능성이 큼
            return {
                "error": "OMR 문항 인식이 불완전합니다(문항 수 부족). "
                         "답안지를 더 정면에서, 종이가 구겨지지 않게 촬영해주세요.",
                "detected_answers_len": len(student_all_ans),
                "rotation_used": chosen_tag
            }

        lc_correct, rc_correct = 0, 0
        part_details = []

        p_defs = [
            ("Part 1", 1, 6),
            ("Part 2", 7, 31),
            ("Part 3", 32, 70),
            ("Part 4", 71, 100),
            ("Part 5", 101, 130),
            ("Part 6", 131, 146),
            ("Part 7", 147, 200),
        ]

        for name, s, e in p_defs:
            p_score, p_items = 0, []
            for i in range(s - 1, e):
                std = student_all_ans[i]
                ans = ANSWER_KEY[i] if i < len(ANSWER_KEY) else "?"
                is_corr = (std == ans)

                if is_corr:
                    p_score += 1
                    if i < 100:
                        lc_correct += 1
                    else:
                        rc_correct += 1

                p_items.append({
                    "no": i + 1,
                    "std": std,
                    "ans": ans,
                    "res": "O" if is_corr else "X"
                })

            part_details.append({
                "name": name,
                "score": p_score,
                "total": e - s + 1,
                "items": p_items
            })

        # 3) (간이 환산) — 실제 TOEIC 환산표가 아니라 네가 쓰던 규칙 유지
        lc_converted = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        rc_converted = min((rc_correct * 5), 495)
        total_converted = lc_converted + rc_converted

        return {
            "rotation_used": chosen_tag,  # 스마트폰 방향 자동 대응 결과
            "lc_correct": lc_correct,
            "rc_correct": rc_correct,
            "lc_converted": lc_converted,
            "rc_converted": rc_converted,
            "total_converted": total_converted,
            "part_details": part_details
        }

    except Exception as e:
        return {"error": str(e)}
