"""
예제 03 — RANSAC 직선 피팅을 직접 구현하기
===========================================

RANSAC 5단계를 코드로 체득한다:
  1. 최소 표본(직선=2점)을 무작위로 뽑는다
  2. 그 점들로 모델(직선)을 세운다
  3. 모든 점의 오차(점-직선 거리)를 계산한다
  4. 오차 <= 임계값인 점 = 인라이어를 센다
  5. 1~4를 반복하여, 인라이어가 가장 많았던 모델을 채택한다

비교를 위해 "최소제곱(LSQ) 피팅"도 함께 그린다.
아웃라이어가 많으면 LSQ는 휘둘리지만 RANSAC은 강건함을 확인하자.
실행하면 output/03_ransac.png 저장.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def make_points(n_inlier=80, n_outlier=60, slope=0.7, intercept=10.0, noise=3.0):
    """직선 위 인라이어 + 무작위 아웃라이어 점 집합 생성."""
    rng = np.random.default_rng(42)
    x_in = rng.uniform(0, 100, n_inlier)
    y_in = slope * x_in + intercept + rng.normal(0, noise, n_inlier)
    x_out = rng.uniform(0, 100, n_outlier)
    y_out = rng.uniform(0, 100, n_outlier)
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    return x, y


def line_from_two_points(p1, p2):
    """두 점으로 직선 ax + by + c = 0 (a^2+b^2=1) 계수를 구한다."""
    (x1, y1), (x2, y2) = p1, p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = np.hypot(a, b)
    if norm < 1e-9:
        return None
    return a / norm, b / norm, c / norm


def point_line_distance(x, y, line):
    """점 (x,y)에서 직선까지의 수직 거리 = 오차 함수."""
    a, b, c = line
    return np.abs(a * x + b * y + c)


def ransac_line(x, y, threshold=5.0, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([x, y])
    n = len(pts)
    best_line, best_inliers = None, None
    best_count = -1

    for _ in range(iters):
        # 1) 최소 표본(2점) 무작위 선택
        i, j = rng.choice(n, size=2, replace=False)
        # 2) 모델(직선) 가설
        line = line_from_two_points(pts[i], pts[j])
        if line is None:
            continue
        # 3) 오차 계산 + 4) 인라이어 선택
        d = point_line_distance(x, y, line)
        inliers = d <= threshold
        count = int(inliers.sum())
        # 5) 가장 인라이어 많은 모델 유지
        if count > best_count:
            best_count = count
            best_line = line
            best_inliers = inliers

    # 마지막으로 인라이어들만으로 최소제곱 재적합(정밀화) — 선택적 단계
    if best_inliers is not None and best_inliers.sum() >= 2:
        xi, yi = x[best_inliers], y[best_inliers]
        A = np.column_stack([xi, np.ones_like(xi)])
        m, b = np.linalg.lstsq(A, yi, rcond=None)[0]
        # y = m x + b  ->  m x - y + b = 0, 정규화(a^2+b^2=1)
        norm = np.hypot(m, -1.0)
        best_line = (m / norm, -1.0 / norm, b / norm)
    return best_line, best_inliers, best_count


def lsq_line(x, y):
    """모든 점을 쓰는 최소제곱 직선 (아웃라이어에 취약)."""
    A = np.column_stack([x, np.ones_like(x)])
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


def main():
    x, y = make_points()
    line, inliers, count = ransac_line(x, y, threshold=5.0, iters=500)
    m_lsq, b_lsq = lsq_line(x, y)

    print(f"RANSAC 인라이어 수: {count} / {len(x)}")
    a, b, c = line
    print(f"RANSAC 직선: {a:.3f}x + {b:.3f}y + {c:.3f} = 0")
    print(f"최소제곱 직선(모든 점): y = {m_lsq:.3f}x + {b_lsq:.3f}  (아웃라이어에 휘둘림)")

    xs = np.linspace(0, 100, 10)
    # RANSAC 직선을 y = f(x) 형태로 변환하여 그리기
    a, b, c = line
    ys_ransac = -(a * xs + c) / b if abs(b) > 1e-6 else np.full_like(xs, np.nan)
    ys_lsq = m_lsq * xs + b_lsq

    plt.figure(figsize=(8, 6))
    plt.scatter(x[~inliers], y[~inliers], c="lightgray", s=20, label="outliers")
    plt.scatter(x[inliers], y[inliers], c="dodgerblue", s=20, label="inliers (RANSAC)")
    plt.plot(xs, ys_ransac, "r-", lw=2.5, label="RANSAC line")
    plt.plot(xs, ys_lsq, "g--", lw=2, label="Least-squares (all points)")
    plt.legend()
    plt.title("RANSAC vs Least-squares line fitting")
    plt.xlim(0, 100); plt.ylim(0, 100)

    out = os.path.join(OUT, "03_ransac.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"저장 완료 -> {out}")


if __name__ == "__main__":
    main()
