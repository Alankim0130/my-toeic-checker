import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 파일 업로드
uploaded = files.upload()
if not uploaded:
    print("파일이 업로드되지 않았습니다.")
else:
    file_name = list(uploaded.keys())[0]

    # 2. 이미지 읽기 및 전처리
    image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 3. 테두리 찾기 (LC, RC 두 구역을 찾습니다)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_regions = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 사각형 형태이고 크기가 충분히 큰 것 2개를 수집
        if len(approx) == 4 and cv2.contourArea(c) > 50000:
            target_regions.append(approx)
            if len(target_regions) == 2: break

    # 찾은 구역을 왼쪽에서 오른쪽 순서로 정렬 (LC -> RC 순서 보장)
    target_regions = sorted(target_regions, key=lambda x: np.mean(x[:, 0, 0]))

    if len(target_regions) > 0:
        total_student_answers = []
        all_debug_imgs = []

        # 4. 각 구역(LC, RC) 루프 실행
        for idx, region in enumerate(target_regions):
            section_name = "LC" if idx == 0 else "RC"
            
            # --- ROI 추출 및 정렬 ---
            pts = region.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
            
            # 강사님의 최적 비율 적용
            dst_w, dst_h = 600, 800 
            dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
            
            # 왼쪽으로 90도 회전 (세로형 정방향)
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = warped.shape[:2]

            # --- 강사님의 황금 좌표 적용 ---
            left_margin = w * 0.082
            top_margin = h * 0.163
            col_spacing = w * 0.201
            row_spacing = h * 0.0422
            bubble_width = w * 0.034

            # RC 영역일 때만 실행되는 보정 로직
            if section_name == "RC":
                  # 1. 시작점(101번)을 아주 살짝 왼쪽으로 당김
                  left_margin = w * 0.080 
                  
                  # 2. 열 간격을 좁혀서 오른쪽으로 갈수록 밀리는 현상 방지
                  # 0.201에서 0.198로 줄여서 5번째 열이 안쪽으로 들어오게 합니다.
                  current_col_spacing = w * 0.196 
                  bubble_width = w * 0.033
            else:
                  # LC는 원래 잘 맞던 수치 그대로 유지
                  left_margin = w * 0.082
                  current_col_spacing = w * 0.201


            # 판독 준비
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(warped_gray, 160, 255, cv2.THRESH_BINARY_INV)
            debug_img = warped.copy()
            labels = ["A", "B", "C", "D"]

            # 100문항 좌표 순회
            for col in range(5):
                for row in range(20):
                    q_idx = (idx * 100) + (col * 20 + row + 1) # 1~100 또는 101~200
                    base_x = left_margin + (col * col_spacing)
                    base_y = top_margin + (row * row_spacing)
                    
                    pixel_counts = []
                    choices_coords = []
                    for j in range(4):
                        cx, cy = int(base_x + (j * bubble_width)), int(base_y)
                        choices_coords.append((cx, cy))
                        
                        mask = np.zeros(warped_gray.shape, dtype="uint8")
                        cv2.circle(mask, (cx, cy), 6, 255, -1)
                        pixel_count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        pixel_counts.append(pixel_count)
                        cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), 1)

                    # 마킹 판독
                    if max(pixel_counts) > 25:
                        total_student_answers.append(labels[np.argmax(pixel_counts)])
                    else:
                        total_student_answers.append("?")
            
            all_debug_imgs.append(debug_img)

        # 5. 결과 시각화 (LC, RC 나란히 표시)
        fig, ax = plt.subplots(1, 2, figsize=(20, 15))
        ax[0].imshow(cv2.cvtColor(all_debug_imgs[0], cv2.COLOR_BGR2RGB)); ax[0].set_title("LC Region")
        if len(all_debug_imgs) > 1:
            ax[1].imshow(cv2.cvtColor(all_debug_imgs[1], cv2.COLOR_BGR2RGB)); ax[1].set_title("RC Region")
        plt.show()

        # 6. 최종 채점 결과 출력
        print("="*50)
        print(f"🎯 통합 채점 완료 (총 {len(total_student_answers)}문항 판독)")
        print("="*50)
        
        for i in range(0, len(total_student_answers), 20):
            section = "LC" if i < 100 else "RC"
            print(f"[{section}] {i+1:3d}~{i+20:3d}번: {' '.join(total_student_answers[i:i+20])}")
            print("-" * 50)
    else:
        print("❌ 테두리를 찾지 못했습니다. LC/RC 구역이 다 나오게 찍어주세요.")

