"""
=====================================================================
  허프 변환(Hough Transform) 직선 검출 - 빈칸 채우기 예제
=====================================================================

[목표]
  이미지에서 '직선'을 찾아내는 허프 변환을 직접 구현해 봅니다.
  TODO(빈칸)만 채우면 완성됩니다. 정답은 hough_answer.py 참고.

[핵심 아이디어]
  하나의 점 (x, y)를 지나는 수많은 직선을 (rho, theta)로 표현합니다.
      rho = x * cos(theta) + y * sin(theta)
  - rho(ρ)   : 원점에서 직선까지의 수직 거리
  - theta(θ) : 그 수직선이 x축과 이루는 각도

  엣지 점 하나하나가 (rho, theta) 공간에 '곡선'을 그리며 투표(vote)하고,
  표를 가장 많이 받은 (rho, theta)가 바로 '직선'입니다.

[실행]
  python hough_exercise.py
=====================================================================
"""

import numpy as np


def hough_lines(edge_image, theta_res=1, rho_res=1, threshold=3):
    """
    edge_image : 엣지 이미지 (0 또는 1). 1인 곳이 엣지 점.
    theta_res  : theta 간격 (도 단위)
    rho_res    : rho 간격 (픽셀 단위)
    threshold  : 이 표수 이상 받은 (rho, theta)만 직선으로 인정

    반환: 검출된 직선들의 리스트 [(rho, theta_deg, votes), ...]
    """
    H, W = edge_image.shape

    # ------------------------------------------------------------------
    # 1) theta 후보 만들기: -90도 ~ +90도 범위
    # ------------------------------------------------------------------
    thetas_deg = np.arange(-90, 90, theta_res)
    thetas_rad = np.deg2rad(thetas_deg)

    # ------------------------------------------------------------------
    # 2) rho 범위 만들기
    #    rho의 최대 절댓값은 이미지 대각선 길이입니다.
    #    TODO(1): 대각선 길이를 구해서 diag_len 에 넣으세요.
    #            힌트: sqrt(가로^2 + 세로^2), np.hypot(W, H) 도 가능
    # ------------------------------------------------------------------
    diag_len = ___________________________  # TODO(1)
    diag_len = int(np.ceil(diag_len))

    # rho 축: -diag_len ~ +diag_len
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)

    # ------------------------------------------------------------------
    # 3) 투표 공간(누산기, accumulator) 만들기 - 전부 0으로 시작
    #    크기: (rho 개수) x (theta 개수)
    # ------------------------------------------------------------------
    accumulator = np.zeros((len(rhos), len(thetas_rad)), dtype=np.int32)

    # ------------------------------------------------------------------
    # 4) 엣지 점들의 좌표 찾기 (값이 1인 위치)
    # ------------------------------------------------------------------
    y_idxs, x_idxs = np.nonzero(edge_image)

    # ------------------------------------------------------------------
    # 5) 투표! 각 엣지 점마다 모든 theta에 대해 rho를 구해 한 표씩 던진다.
    # ------------------------------------------------------------------
    for x, y in zip(x_idxs, y_idxs):
        for t_idx, theta in enumerate(thetas_rad):
            # TODO(2): 허프 변환 공식으로 rho 계산
            #          rho = x*cos(theta) + y*sin(theta)
            rho = ___________________________  # TODO(2)

            # rho 값을 accumulator의 '행 번호'로 바꾸기
            # rhos 배열이 -diag_len 부터 시작하므로 diag_len 을 더해 0부터 시작하게 맞춤
            rho_idx = int(round(rho)) + diag_len

            # TODO(3): 해당 칸에 한 표를 더하세요.
            ___________________________  # TODO(3)

    # ------------------------------------------------------------------
    # 6) threshold 이상 득표한 칸을 직선으로 모으기
    # ------------------------------------------------------------------
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            votes = accumulator[rho_idx, t_idx]

            # TODO(4): votes가 threshold 이상일 때만 직선으로 인정하는 조건
            if ___________________________:  # TODO(4)
                lines.append((rhos[rho_idx], thetas_deg[t_idx], int(votes)))

    return lines


# =====================================================================
#  테스트: 대각선 모양의 점들을 만들어 직선이 검출되는지 확인
# =====================================================================
if __name__ == "__main__":
    # 7x7 이미지에 대각선(y = x) 위의 점들을 1로 찍는다.
    img = np.zeros((7, 7), dtype=np.uint8)
    for i in range(7):
        img[i, i] = 1   # (0,0),(1,1),...,(6,6) → 대각선

    print("입력 엣지 이미지:")
    print(img)

    lines = hough_lines(img, theta_res=1, rho_res=1, threshold=5)

    print(f"\n검출된 직선 개수: {len(lines)}")
    # 표를 가장 많이 받은 순으로 정렬해 상위 3개 출력
    lines.sort(key=lambda L: L[2], reverse=True)
    print("rho    theta(도)   votes")
    for rho, theta, votes in lines[:3]:
        print(f"{rho:>5}   {theta:>6}     {votes}")

    # 기대 결과: 대각선(y=x)이면 theta가 -45도 근처, rho는 0 근처에서
    #            표를 가장 많이(7표 가까이) 받아야 합니다.
