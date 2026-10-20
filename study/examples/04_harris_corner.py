"""
예제 04 — 해리스 코너 검출을 numpy로 직접 구현하기
===================================================

해리스 코너 공식을 코드로 확인한다:
  1. 영상 미분 Ix, Iy 계산
  2. 2차 모멘트 행렬 성분 Ix^2, Iy^2, Ix*Iy 를 가우시안으로 가중 합산
       M = [ Sxx  Sxy ]
           [ Sxy  Syy ]
  3. 코너 응답:  R = det(M) - alpha * (trace(M))^2
                   = (Sxx*Syy - Sxy^2) - alpha*(Sxx + Syy)^2
  4. R > threshold 인 점만 남김
  5. 비최대 억제(지역 최대)로 대표 코너 선택

alpha 는 보통 0.04 ~ 0.06.
실행하면 output/04_harris.png 에 (1)입력, (2)R 응답맵, (3)검출 코너 저장.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def make_image(h=120, w=120):
    """코너가 뚜렷한 합성 영상: 흰 배경에 검은 사각형 + 삼각형."""
    img = np.full((h, w), 1.0)
    img[25:70, 25:70] = 0.0            # 사각형 → 4개 코너
    for i in range(40):                # 삼각형 (대각 에지 + 꼭짓점)
        img[80 + 0: 80 + i, 80: 80 + i] = 0.0
    return img


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def conv2d(img, kernel):
    """간단한 2D 상관(에지는 reflect 패딩)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kernel)
    return out


def harris_response(img, alpha=0.04, sigma=1.0):
    # 1) 영상 미분 (Sobel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    Ix = conv2d(img, sobel_x)
    Iy = conv2d(img, sobel_y)

    # 2) 2차 모멘트 성분 + 가우시안 가중 합산
    g = gaussian_kernel(5, sigma)
    Sxx = conv2d(Ix * Ix, g)
    Syy = conv2d(Iy * Iy, g)
    Sxy = conv2d(Ix * Iy, g)

    # 3) 코너 응답 R = det(M) - alpha*trace(M)^2
    detM = Sxx * Syy - Sxy * Sxy
    traceM = Sxx + Syy
    R = detM - alpha * (traceM ** 2)
    return R


def non_max_suppression(R, thresh_ratio=0.01, win=5):
    """R > (최댓값*비율) 이고 지역 최대인 점만 코너로 선택."""
    thresh = thresh_ratio * R.max()
    corners = []
    h, w = R.shape
    r = win // 2
    for y in range(r, h - r):
        for x in range(r, w - r):
            val = R[y, x]
            if val <= thresh:
                continue
            local = R[y - r:y + r + 1, x - r:x + r + 1]
            if val >= local.max():   # 지역 최대
                corners.append((x, y))
    return corners


def main():
    img = make_image()
    R = harris_response(img, alpha=0.04, sigma=1.0)
    corners = non_max_suppression(R, thresh_ratio=0.02, win=7)
    print(f"검출된 코너 개수: {len(corners)}")
    print("좌표 (x, y):", corners)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("1) Input image")

    im = axes[1].imshow(R, cmap="jet")
    axes[1].set_title("2) Harris response R\n(R>0 corner, R<0 edge)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(img, cmap="gray")
    if corners:
        cx, cy = zip(*corners)
        axes[2].scatter(cx, cy, c="red", s=60, marker="x")
    axes[2].set_title("3) Detected corners\n(after non-max suppression)")

    out = os.path.join(OUT, "04_harris.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"저장 완료 -> {out}")


if __name__ == "__main__":
    main()
