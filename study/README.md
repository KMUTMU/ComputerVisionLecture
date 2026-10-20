# 컴퓨터비전 6·7주차 시험 공부 자료

이 폴더는 **6주차(선분 검출: 허프 변환 · RANSAC)** 와 **7주차(지역 특징 검출: 해리스 코너 등)** 강의 내용을
시험 대비용으로 쉽고 자세하게 정리한 자료입니다. 핵심 개념 설명 + 직접 돌려볼 수 있는 파이썬 예제가 함께 있습니다.

## 📂 폴더 구성

```
study/
├── README.md                     # 이 파일 (전체 안내)
├── 6주차_선분검출.md              # 허프 변환, RANSAC 핵심 정리
├── 7주차_특징검출.md              # 해리스 코너, 헤시안, LoG, SUSAN 정리
├── 시험_예상문제.md               # 예상 문제 + 모범 답안
└── examples/                     # 실행 가능한 파이썬 예제
    ├── requirements.txt
    ├── 01_hough_line_numpy.py    # 허프 변환을 numpy로 직접 구현
    ├── 02_hough_opencv.py        # OpenCV HoughLines 사용 예
    ├── 03_ransac_line.py         # RANSAC 직선 피팅 직접 구현
    ├── 04_harris_corner.py       # 해리스 코너를 numpy로 직접 구현
    └── 05_harris_opencv.py       # OpenCV cornerHarris 사용 예
```

## 🚀 예제 실행 방법

```bash
cd study/examples
pip install -r requirements.txt
python 01_hough_line_numpy.py
```

각 예제는 결과 이미지를 `examples/output/` 폴더에 저장합니다.
(OpenCV 창을 띄우지 않고 파일로 저장하므로 어디서든 실행 가능합니다.)

## 📝 공부 순서 추천

1. `6주차_선분검출.md` → 허프 변환의 "투표(voting)" 개념부터 이해
2. `examples/01_hough_line_numpy.py` 실행하며 누적 배열이 어떻게 쌓이는지 확인
3. `examples/03_ransac_line.py`로 RANSAC의 "뽑고–맞춰보고–반복" 흐름 체득
4. `7주차_특징검출.md` → 해리스 코너의 2차 모멘트 행렬과 고유값 의미 이해
5. `examples/04_harris_corner.py`로 코너 응답 함수 R이 어떻게 계산되는지 확인
6. `시험_예상문제.md`로 마무리 점검
```
```

> 자료는 강의 슬라이드 내용을 바탕으로 재구성했습니다.
