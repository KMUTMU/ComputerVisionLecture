"""
예제 05 — OpenCV cornerHarris 사용하기
=======================================

실무에서는 직접 구현(예제 04) 대신 cv2.cornerHarris 를 씁니다.
  cv2.cornerHarris(gray, blockSize, ksize, k)
    - blockSize : 2차 모멘트 행렬 M을 계산할 이웃 창 크기
    - ksize     : Sobel 미분 커널 크기
    - k         : 코너 응답식의 alpha (보통 0.04 ~ 0.06)

반환값은 각 화소의 코너 응답 R. R > 임계값 인 점을 코너로 표시한다.
실행하면 output/05_harris_opencv.png 저장.
"""
import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python 이 필요합니다:  pip install opencv-python")

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def make_image(h=200, w=200):
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (120, 120), 0, -1)       # 채운 사각형(코너 4개)
    cv2.rectangle(img, (130, 130), (180, 180), 0, 3)      # 테두리 사각형
    pts = np.array([[150, 30], [190, 90], [110, 90]], np.int32)
    cv2.fillPoly(img, [pts], 0)                            # 삼각형(꼭짓점 3개)
    return img


def main():
    gray = make_image()
    gray_f = np.float32(gray)

    # 해리스 코너 응답 R
    R = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    R = cv2.dilate(R, None)  # 코너 표시를 보기 좋게 키움

    out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # R > 임계값 인 화소를 빨간색으로
    mask = R > 0.01 * R.max()
    out_img[mask] = [0, 0, 255]
    print(f"코너로 표시된 화소 수: {int(mask.sum())}")

    # cv2.goodFeaturesToTrack = 해리스/Shi-Tomasi 기반 대표 코너점 추출(비최대 억제 포함)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.1,
                                      minDistance=10)
    out_img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for c in corners.astype(int).reshape(-1, 2):
            cv2.circle(out_img2, tuple(c), 4, (0, 180, 0), 2)
        print(f"goodFeaturesToTrack 대표 코너 개수: {len(corners)}")

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([gray_bgr, out_img, out_img2])
    out = os.path.join(OUT, "05_harris_opencv.png")
    cv2.imwrite(out, canvas)
    print(f"저장 완료 -> {out}  (좌:입력, 중:cornerHarris, 우:goodFeaturesToTrack)")


if __name__ == "__main__":
    main()
