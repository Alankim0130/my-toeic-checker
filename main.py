import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import time
import uuid

ANSWERS_DB = {
    "vol16": ["C","C","D","A","C","A","B","A","C","A","C","C","A","C","A","B","B","C","A","C","A","A","A","C","A","A","B","A","B","A","A","C","B","A","B","B","A","C","D","C","B","D","C","C","D","A","A","C","B","C","C","A","D","C","D","C","B","D","C","A","A","B","C","A","C","B","C","B","D","D","D","C","A","C","B","D","C","B","A","D","B","B","B","A","B","B","D","A","B","A","D","C","C","D","C","A","C","D","C","A","B","B","A","A","A","D","C","B","C","B","C","D","B","B","D","A","D","B","D","B","C","C","D","D","A","C","C","D","D","D","C","B","A","D","D","B","C","A","B","D","A","D","C","B","A","A","A","C","A","A","A","D","D","A","B","D","C","A","B","C","B","C","A","D","D","C","D","D","A","A","A","C","D","D","A","B","A","C","C","D","C","B","C","B","C","D","C","A","B","D","B","A","A","B","D","C","A","B","B","D"],
    "vol17": ["D","A","A","B","C","A","C","B","C","B","C","B","B","C","B","A","B","B","A","B","B","B","C","B","A","C","B","A","B","B","A","C","B","A","D","A","A","B","C","D","B","C","A","A","D","C","D","B","C","B","A","C","A","B","B","C","B","D","D","C","A","A","D","D","D","C","A","B","A","B","A","C","B","A","B","C","D","C","B","D","C","A","B","C","A","D","C","C","B","C","D","D","C","A","C","B","D","D","C","C","C","A","D","C","A","A","B","D","B","D","B","A","C","B","D","A","B","C","A","A","B","D","B","C","B","C","A","D","C","C","A","A","D","D","A","B","B","C","B","C","A","C","D","C","D","C","D","B","D","A","D","A","D","C","C","A","C","C","D","B","C","B","D","A","C","D","B","C","D","C","A","C","B","A","B","B","D","B","A","C","B","D","D","C","C","A","C","A","B","D","A","B","B","A","D","C","A","C","B","A"]
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ====== Debug storage (동시 사용자 안전) ======
DEBUG_DIR = "debug"
DEBUG_TTL_SECONDS = 60 * 10  # 10분 후 자동 정리 (원하면 늘리기)

os.makedirs(DEBUG_DIR, exist_ok=True)

def cleanup_old_debug_files():
    """TTL 지난 디버그 파일 자동 삭제 (간단 정리)"""
    now = time.time()
    try:
        for fn in os.listdir(DEBUG_DIR):
            if not fn.lower().endswith(".jpg"):
                continue
            path = os.path.join(DEBUG_DIR, fn)
            try:
                if now - os.path.getmtime(path) > DEBUG_TTL_SECONDS:
                    os.remove(path)
            except Exception:
                pass
    except Exception:
        pass

# ====== Geometry utils ======
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def preprocess_for_detection(bgr: np.ndarray) -> np.ndarray:
    """스마트폰 그림자 대응 전처리: 조명 평탄화 + CLAHE"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = 61
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    return norm

def find_omr_regions(gray_norm: np.ndarray) -> list:
    edged = cv2.Canny(gray_norm, 40, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray_norm.shape[:2]
    img_area = h * w

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.03:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            pts = np.array(cv2.boxPoints(cv2.minAreaRect(c)), dtype="float32")

        rect = order_points(pts)

        (tl, tr, br, bl) = rect
        ww = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        hh = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        if ww <= 1 or hh <= 1:
            continue
        ar = ww / hh
        if ar < 1.0 or ar > 3.2:
            continue

        candidates.append((area, rect))

    candidates.sort(key=lambda x: x[0], reverse=True)

    regions = []
    for _, rect in candidates:
        if len(regions) == 0:
            regions.append(rect)
        else:
            c1 = rect.mean(axis=0)
            c0 = regions[0].mean(axis=0)
            if np.linalg.norm(c1 - c0) < min(w, h) * 0.15:
                continue
            regions.append(rect)
        if len(regions) == 2:
            break

    regions.sort(key=lambda r: r[:, 0].mean())
    return regions

def warp_region(gray_norm: np.ndarray, bgr: np.ndarray, rect: np.ndarray, dst_w=800, dst_h=600):
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_gray = cv2.warpPerspective(gray_norm, M, (dst_w, dst_h))
    warped_bgr = cv2.warpPerspective(bgr, M, (dst_w, dst_h))
    return warped_gray, warped_bgr

def make_thresh_for_marks(warped_gray: np.ndarray) -> np.ndarray:
    h, w = warped_gray.shape[:2]
    block = int(max(51, (min(w, h) // 10) | 1))
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
    best = int(np.max(counts))
    idx = int(np.argmax(counts))
    sorted_counts = sorted([int(x) for x in counts])
    second = sorted_counts[-2] if len(sorted_counts) >= 2 else 0
    if best >= min_pixels and (best / (second + 1)) >= ratio:
        return idx
    return None

# ====== Debug image API (debug_id 기반) ======
@app.get("/get-debug-image/{debug_id}")
def get_debug_image(debug_id: str):
    path = os.path.join(DEBUG_DIR, f"debug_{debug_id}.jpg")
    if not os.path.exists(path):
        return JSONResponse(
            status_code=404,
            content={"error": "디버그 이미지가 없습니다(만료/삭제/ID 오류). 다시 채점 후 확인해 주세요."}
        )
    cleanup_old_debug_files()
    return FileResponse(path, media_type="image/jpeg", filename=f"debug_{debug_id}.jpg")

# ====== Main analyze ======
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    vol: str = Form(None),
    debug: int = Form(1)  # 학생용도 링크가 필요하면 1 유지 / 끄려면 0
):
    try:
        cleanup_old_debug_files()

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다. 다른 사진으로 다시 시도해주세요."}

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
                         "답안지를 화면에 꽉 차게, 밝은 곳에서, 흔들림 없이 촬영해주세요."
            }

        labels = ["A", "B", "C", "D"]
        student_all_ans = []
        debug_views = []

        for idx, rect in enumerate(regions):
            warped_gray, warped_bgr = warp_region(chosen_gray, chosen_bgr, rect, dst_w=800, dst_h=600)
            th = make_thresh_for_marks(warped_gray)

            vis = warped_bgr.copy()

            dst_w, dst_h = 800, 600
            l_margin = dst_w * 0.0842
            t_margin = dst_h * 0.1635
            c_gap = dst_w * 0.1988
            r_gap = dst_h * 0.0421
            b_w = dst_w * 0.0342

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
                            cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

                    choice = pick_choice_from_counts(counts, min_pixels=22, ratio=1.25)

                    if choice is None:
                        student_all_ans.append("?")
                    else:
                        student_all_ans.append(labels[choice])
                        if debug:
                            sx, sy = centers[choice]
                            cv2.circle(vis, (sx, sy), 8, (0, 255, 0), 2)

            if debug:
                debug_views.append(vis)

        # debug 저장 (요청마다 파일명 다르게)
        debug_id = None
        debug_url = None
        if debug and len(debug_views) == 2:
            debug_id = uuid.uuid4().hex
            debug_path = os.path.join(DEBUG_DIR, f"debug_{debug_id}.jpg")
            cv2.imwrite(debug_path, np.hstack(debug_views))
            # 같은 도메인 기준 상대 URL
            debug_url = f"/get-debug-image/{debug_id}"

        clean_vol = (vol.replace(".", "") if vol else "vol16").strip()
        ANSWER_KEY = ANSWERS_DB.get(clean_vol, ANSWERS_DB["vol16"])

        if len(student_all_ans) < 200:
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

                p_items.append({"no": i + 1, "std": std, "ans": ans, "res": "O" if is_corr else "X"})

            part_details.append({"name": name, "score": p_score, "total": e - s + 1, "items": p_items})

        lc_converted = min((lc_correct * 5) + 10, 495) if lc_correct > 0 else 0
        rc_converted = min((rc_correct * 5), 495)
        total_converted = lc_converted + rc_converted

        return {
            "rotation_used": chosen_tag,
            "lc_correct": lc_correct,
            "rc_correct": rc_correct,
            "lc_converted": lc_converted,
            "rc_converted": rc_converted,
            "total_converted": total_converted,
            "part_details": part_details,
            # ✅ 디버그 링크 정보 (프론트가 이걸로 URL 만들면 됨)
            "debug_id": debug_id,
            "debug_url": debug_url
        }

    except Exception as e:
        return {"error": str(e)}
