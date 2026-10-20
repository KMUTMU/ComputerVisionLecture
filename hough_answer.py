"""
=====================================================================
  허프 변환(Hough Transform) 직선 검출 - 정답 (완성본)
=====================================================================
  hough_exercise.py 의 TODO(1)~(4)를 모두 채운 완성 버전입니다.
  빈칸 정답은 아래 주석 ===> 표시를 참고하세요.

  실행: python hough_answer.py
=====================================================================
"""

import numpy as np


def hough_lines(edge_image, theta_res=1, rho_res=1, threshold=3):
    H, W = edge_image.shape

    # 1) theta 후보: -90 ~ +90도
    thetas_deg = np.arange(-90, 90, theta_res)
    thetas_rad = np.deg2rad(thetas_deg)

    # 2) rho 범위 = 이미지 대각선 길이
    # ===> TODO(1) 정답: 대각선 = sqrt(W^2 + H^2)
    diag_len = np.hypot(W, H)        # == np.sqrt(W**2 + H**2)
    diag_len = int(np.ceil(diag_len))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)

    # 3) 누산기(투표 공간) 0으로 초기화
    accumulator = np.zeros((len(rhos), len(thetas_rad)), dtype=np.int32)

    # 4) 엣지 점 좌표
    y_idxs, x_idxs = np.nonzero(edge_image)

    # 5) 투표
    for x, y in zip(x_idxs, y_idxs):
        for t_idx, theta in enumerate(thetas_rad):
            # ===> TODO(2) 정답: 허프 변환 공식
            rho = x * np.cos(theta) + y * np.sin(theta)

            rho_idx = int(round(rho)) + diag_len

            # ===> TODO(3) 정답: 해당 칸에 한 표 추가
            accumulator[rho_idx, t_idx] += 1

    # 6) threshold 이상만 직선으로
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            votes = accumulator[rho_idx, t_idx]
            # ===> TODO(4) 정답: 임계값 이상인지 확인
            if votes >= threshold:
                lines.append((rhos[rho_idx], thetas_deg[t_idx], int(votes)))

    return lines


if __name__ == "__main__":
    img = np.zeros((7, 7), dtype=np.uint8)
    for i in range(7):
        img[i, i] = 1

    print("입력 엣지 이미지:")
    print(img)

    lines = hough_lines(img, theta_res=1, rho_res=1, threshold=5)

    print(f"\n검출된 직선 개수: {len(lines)}")
    lines.sort(key=lambda L: L[2], reverse=True)
    print("rho    theta(도)   votes")
    for rho, theta, votes in lines[:3]:
        print(f"{rho:>5}   {theta:>6}     {votes}")

    print("\n[해설]")
    print("대각선 y=x 위의 점 7개가 모두 같은 (rho, theta)에 투표하여")
    print("theta=-45도, rho=0 근처에서 가장 많은 표(7표)를 받습니다.")
    print("이것이 바로 검출된 직선입니다.")
    print()
    print("[기대 출력 예]")
    print("  rho=0   theta=-48~-46   votes=7")
    print("  (정확히 -45가 아닌 건 rho를 정수로 반올림하는 양자화 때문. 원리는 동일)")
