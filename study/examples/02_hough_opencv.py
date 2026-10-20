"""
예제 02 — OpenCV HoughLines / HoughLinesP 사용하기
==================================================

직접 구현(예제 01)과 달리, 실무에서는 OpenCV의 내장 함수를 씁니다.
  - cv2.HoughLines    : 표준 허프 변환, (rho, theta) 반환
  - cv2.HoughLinesP   : 확률적 허프 변환, 선분의 양 끝점 (x1,y1,x2,y2) 반환

시험 포인트: 입력은 보통 Canny 에지 영상이어야 한다 (에지 점에 투표하므로).
실행하면 output/02_hough_opencv.png 저장.
"""
import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python 이 필요합니다:  pip install opencv-python")

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def make_test_image(h=200, w=300):
    """흰 배경에 몇 개의 직선/사각형을 그린 합성 영상."""
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.line(img, (20, 20), (280, 60), 0, 2)     # 비스듬한 직선
    cv2.line(img, (10, 180), (290, 120), 0, 2)   # 또 다른 직선
    cv2.rectangle(img, (120, 90), (200, 160), 0, 2)  # 사각형(수직/수평 선)
    return img


def main():
    gray = make_test_image()

    # 1) 에지 검출 (허프의 입력)
    edges = cv2.Canny(gray, 50, 150)

    # 2) 표준 허프 변환
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=80)
    color_std = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:, 0][:12]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(color_std, p1, p2, (0, 0, 255), 1)
    print(f"HoughLines 검출 개수: {0 if lines is None else len(lines)}")

    # 3) 확률적 허프 변환 (선분의 끝점)
    linesP = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=50, minLineLength=40, maxLineGap=10)
    color_p = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:, 0]:
            cv2.line(color_p, (x1, y1), (x2, y2), (0, 180, 0), 2)
    print(f"HoughLinesP 검출 선분 개수: {0 if linesP is None else len(linesP)}")

    # 저장 (좌: 에지, 중: 표준 허프, 우: 확률적 허프)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([edges_bgr, color_std, color_p])
    out = os.path.join(OUT, "02_hough_opencv.png")
    cv2.imwrite(out, canvas)
    print(f"저장 완료 -> {out}  (좌:Canny에지, 중:HoughLines, 우:HoughLinesP)")


if __name__ == "__main__":
    main()
