"""
예제 01 — 허프 변환(직선) 을 numpy로 직접 구현하기
===================================================

목표: "투표(voting)" 개념을 눈으로 확인한다.
  - 영상의 각 에지 점이 (rho, theta) 파라미터 공간의 여러 칸에 한 표씩 던진다.
  - 같은 직선 위의 점들이 던진 표는 한 칸에 몰린다 → 그 칸이 검출된 직선.

극좌표(normal form) 직선식:   rho = x*cos(theta) + y*sin(theta)
  - theta: 직선에 수직인 방향의 각도 (0 ~ 180도)
  - rho  : 원점에서 직선까지의 수직 거리

이 예제는 OpenCV 없이 numpy만으로 누적 배열(accumulator)을 직접 쌓습니다.
실행하면 output/01_hough.png 에 (1) 입력 에지, (2) 누적 배열, (3) 검출된 직선을 저장합니다.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def make_edge_image(h=100, w=100):
    """노이즈가 섞인, 두 개의 직선으로 이루어진 에지 점 영상을 생성."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w), dtype=np.uint8)

    # 직선 1:  y = 0.5*x + 10
    xs = np.arange(0, w)
    ys = (0.5 * xs + 10).astype(int)
    for x, y in zip(xs, ys):
        if 0 <= y < h:
            img[y, x] = 1

    # 직선 2:  거의 수직에 가까운 직선 x = 70 (수직선도 극좌표라 표현 가능!)
    for y in range(h):
        img[y, 70] = 1

    # 무작위 노이즈 점 (아웃라이어)
    for _ in range(60):
        img[rng.integers(0, h), rng.integers(0, w)] = 1
    return img


def hough_lines(edge_img, theta_res_deg=1.0, rho_res=1.0):
    """numpy로 직선 허프 변환. 누적 배열과 축 정보를 반환."""
    h, w = edge_img.shape
    # theta: 0 ~ 180도
    thetas = np.deg2rad(np.arange(0.0, 180.0, theta_res_deg))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # rho 범위: -대각선 ~ +대각선
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1, rho_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(edge_img)  # 에지 점 좌표
    for x, y in zip(xs, ys):
        # 각 에지 점은 모든 theta에 대해 rho를 계산하고 그 칸에 +1 투표
        rho_vals = x * cos_t + y * sin_t
        rho_idx = np.round((rho_vals + diag) / rho_res).astype(int)
        for ti, ri in enumerate(rho_idx):
            if 0 <= ri < len(rhos):
                accumulator[ri, ti] += 1
    return accumulator, rhos, thetas


def find_peaks(accumulator, rhos, thetas, num=2, min_dist=10):
    """누적 배열에서 표를 많이 받은 상위 num개 peak(=직선)를 찾는다."""
    acc = accumulator.copy()
    peaks = []
    for _ in range(num):
        idx = np.argmax(acc)
        ri, ti = np.unravel_index(idx, acc.shape)
        peaks.append((rhos[ri], thetas[ti], acc[ri, ti]))
        # 주변을 지워 가까운 중복 peak 방지 (간단한 비최대 억제)
        r0, r1 = max(0, ri - min_dist), min(acc.shape[0], ri + min_dist)
        t0, t1 = max(0, ti - min_dist), min(acc.shape[1], ti + min_dist)
        acc[r0:r1, t0:t1] = 0
    return peaks


def main():
    edge = make_edge_image()
    acc, rhos, thetas = hough_lines(edge)
    peaks = find_peaks(acc, rhos, thetas, num=2)

    print("검출된 직선 (rho, theta[deg], 표 수):")
    for rho, theta, votes in peaks:
        print(f"  rho={rho:7.1f}, theta={np.rad2deg(theta):6.1f}deg, votes={votes}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(edge, cmap="gray")
    axes[0].set_title("1) Input edge points")

    axes[1].imshow(acc, cmap="hot", aspect="auto",
                   extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]),
                           rhos[-1], rhos[0]])
    axes[1].set_title("2) Hough accumulator (votes)")
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("rho (px)")

    axes[2].imshow(edge, cmap="gray")
    h, w = edge.shape
    for rho, theta, _ in peaks:
        # 극좌표 직선을 영상 위에 그리기
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pts = np.array([[x0 + 1000 * (-b), y0 + 1000 * a],
                        [x0 - 1000 * (-b), y0 - 1000 * a]])
        axes[2].plot(pts[:, 0], pts[:, 1], "r-", linewidth=1.5)
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)
    axes[2].set_title("3) Detected lines")

    out = os.path.join(OUT, "01_hough.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\n저장 완료 -> {out}")


if __name__ == "__main__":
    main()
