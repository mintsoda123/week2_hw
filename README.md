# week2_submission
202312143 이민영 week2에 대한 과제 제출입니다. 
01~04의 방법들을 이용해서 훅의 법칙에 관한 웹사이트를 만들었습니다.

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | Tailwind CSS, KaTeX |
| 시각화 | Matplotlib → PNG |
| ML / 수치계산 | TensorFlow, NumPy |

---

# 01. 훅의 법칙 × TensorFlow

**폴더:** `LinRegSpr/` | **포트:** `localhost:8000`

> 용수철에 질량을 달면 얼마나 늘어나는가? — AI가 물리 법칙을 스스로 학습

훅의 법칙 `F = kx`를 TensorFlow의 가장 단순한 신경망(`Dense(1)`)으로 학습합니다. 이론값 k=2, L₀=10에 AI가 스스로 수렴해가는 과정을 Epoch별 Loss 그래프로 확인할 수 있습니다.

sklearn 등 외부 ML 라이브러리 없이 **TensorFlow + NumPy만으로** 직접 구현했으며, 60개 샘플 데이터로 R² > 0.98 달성을 목표로 자동 재시도 학습을 수행합니다.


---

## 알고리즘 흐름

```
1. 데이터 생성  →  k=2 cm/kg, L₀=10 cm 기준 60개 샘플 (σ=0.45 노이즈 추가)
2. 모델 구성   →  Dense(1) 단층 신경망, Adam lr=0.08
3. 반복 학습   →  R² < 0.98이면 Epoch × 1.6 늘려서 최대 12회 재시도
4. 시각화      →  회귀선 / Loss 곡선 / 잔차 / Loss Landscape PNG 저장
```

---

## 기능

- Epoch 수 슬라이더 조절 (100~2000)
- 질량(kg) 입력 → 용수철 길이 예측 + 이론값 오차(%) 출력
- 학습 결과: R² / 학습된 k, b / 목표 달성 여부 표시
- Loss 곡선 / 회귀선 / 잔차 분석 / Loss Landscape PNG 저장

---

## 핵심 개념

| 개념 | 설명 |
|---|---|
| 선형회귀 | `y = wx + b` 형태로 데이터를 직선으로 근사 |
| Dense(1) | 입력 1개 → 출력 1개 선형 변환층 |
| MSE Loss | (예측 - 실제)² 평균, 작을수록 좋음 |
| Adam | 학습률 자동 조정 경사하강법 |
| R² Score | 예측 정확도 지표 (목표: 0.98 이상) |
| Loss Landscape | k, b 파라미터 공간에서의 손실 지형도 |

---

## 프로젝트 구조

```
LinRegSpr/
├── main.py             # FastAPI 서버 (학습 / 예측 API)
├── train_model.py      # TensorFlow 모델 학습 및 시각화
├── requirements.txt    # 의존성 패키지
├── static/
│   └── index.html      # 프론트엔드 UI
└── output/             # 생성된 PNG 저장 디렉토리
    ├── spring_fitting.png
    ├── loss_curve.png
    ├── residuals.png
    └── loss_landscape.png
```

---

## 서버 실행

```bash
pip install -r requirements.txt
python week2/LinRegSpr/main.py
```

또는

```bash
uvicorn main:app --reload --port 8000
```

---

# 02. K-Means 군집화 × NumPy

**폴더:** `UnSupClust/` | **포트:** `localhost:8001`

> 정답(레이블) 없이 데이터 스스로 3개 그룹을 찾아가는 과정

sklearn 등 외부 ML 라이브러리를 일절 사용하지 않고, **numpy 수식만으로** K-Means를 처음부터 구현했습니다. 90개 2D 데이터가 초기화 → 배정 → 이동 → 수렴 4단계를 반복하며 군집을 찾는 전 과정을 시각화합니다.

---

## 알고리즘 4단계

```
1. 초기화  →  데이터에서 K개(=3) 랜덤 중심점 선택
2. 배정    →  각 점을 가장 가까운 중심점에 배정 (유클리드 거리)
3. 이동    →  각 클러스터의 평균으로 중심점 갱신
4. 수렴    →  중심점 변화량 < 1e-6 이면 조기 종료
```

---

## 기능

- 최대 반복 횟수 슬라이더 (1~10회)
- (x, y) 입력 → 가장 가까운 클러스터 예측
- 군집 결과 산점도(★ 중심점) / WCSS Loss 곡선 / 예측 결과 PNG 저장

---

## 핵심 개념

| 개념 | 설명 |
|---|---|
| 비지도 학습 | 정답 없이 데이터 구조만으로 패턴 발견 |
| WCSS | Within-Cluster Sum of Squares, 군집 밀집도 지표 |
| 유클리드 거리 | 두 점 사이 직선 거리로 가장 가까운 클러스터 결정 |

---

## 서버 실행

```bash
python week2/UnSupClust/main.py
```
---

# 03. 데이터 전처리 × Min-Max Normalization
**폴더: `Data_Pre/` | 포트: `localhost:8002`**

> 단위가 다른 물리 데이터를 0~1로 압축하면 TensorFlow 학습이 어떻게 달라지는가?

질량(0.1 ~ 10 kg)과 변위(0.01 ~ 0.98 m)처럼 스케일이 다른 훅의 법칙 데이터를 sklearn 없이 **Min-Max 정규화 공식 하나로 직접 변환**합니다. 정규화 전/후를 비교하고, TensorFlow MLP로 학습하여 새로운 질량의 용수철 변위를 예측합니다.

---

## 정규화 공식

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

결과값은 항상 **0 ≤ x' ≤ 1** 범위에 들어옵니다.

---

## 왜 정규화가 필요한가?

| 상황 | 문제 |
|------|------|
| **정규화 없이 학습** | 질량(kg)과 변위(m)의 스케일 차이로 gradient 불안정 |
| **정규화 후 학습** | 두 특성 모두 0~1 동일 스케일로 수렴 안정화 |

---

## 알고리즘

훅의 법칙: **F = k · x** → **x = (m · g) / k**

| 기호 | 설명 | 값 |
|------|------|----|
| m | 질량 (kg) | 0.1 ~ 10.0 |
| g | 중력 가속도 | 9.8 m/s² |
| k | 용수철 상수 | 10 N/m |
| x | 변위 (m) | 예측 대상 |

---

## 기능

- Epochs / Learning Rate 슬라이더로 하이퍼파라미터 조정
- 질량 입력 → 용수철 변위 예측 (cm / m)
- 정규화 전/후 산점도 / Loss 곡선 / 회귀 피팅 / 예측 결과 PNG 저장

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Min-Max Normalization** | 최솟값→0, 최댓값→1로 선형 변환, 역변환으로 실제 단위 복원 |
| **Feature Scaling** | 특성 간 단위 차이를 제거해 ML 학습 안정화 |
| **TensorFlow MLP** | Dense(32→16→8→1) 다층 퍼셉트론으로 회귀 학습 |

---

## 출력 파일 (`output/`)

| 파일 | 내용 |
|------|------|
| `01_normalization_comparison.png` | 정규화 전/후 산점도 비교 |
| `02_loss_curve.png` | Epoch별 MSE Loss 곡선 (log scale) |
| `03_regression_fit.png` | 실제 데이터 + TF 회귀선 + 이론값 비교 |
| `04_prediction_result.png` | 예측 결과 crosshair 시각화 |

---

## 서버 실행

```bash
python week2/Data_Pre/main.py
```
---

# 04. 경사 하강법 × NumPy + TensorFlow
**폴더:** `week2/Gre_Des_Vis/` | **포트:** `localhost:8003`

> 산에서 공이 굴러 내려가듯, 기울기의 반대 방향으로 이동해 최솟값을 찾는다

손실 함수 `f(x) = x²`에서 경사 하강법이 최솟값 `x=0`을 향해 이동하는 과정을 시각화합니다.
학습률(lr)이 너무 크면 발산하는 것도 직접 체험할 수 있어 머신러닝의 핵심 최적화 개념을 가장 직관적으로 전달합니다.
나아가 이 원리를 **훅의 법칙(Hooke's Law)** 에 적용하여 TensorFlow가 스프링 상수 `k`를 학습하는 과정까지 시각화합니다.

---

## 업데이트 규칙

$$x_{t+1} = x_t - \alpha \cdot \nabla f(x_t) = x_t - \alpha \cdot 2x_t$$

---

## 학습률에 따른 동작 차이

| 학습률 (α) | 동작 | 결과 |
|-----------|------|------|
| 0.01 ~ 0.05 | 매우 조금씩 이동 | 수렴하지만 느림 |
| 0.1 ~ 0.4 | 적당히 이동 | 빠르고 안정적으로 수렴 ✓ |
| 0.8 ~ 0.9 | 최솟값을 넘어 진동 | 진동하며 느리게 수렴 |
| ≥ 1.0 | 반대 방향으로 튕김 | 발산 💥 |

---

## 기능

### ⚡ Tab 1 — 경사 하강법 시각화
- 시작점 x₀ 슬라이더 (-5 ~ 5)
- 학습률 α 슬라이더 (0.01 ~ 0.95) — 위험 구간 색상 경고
- 스텝 수 슬라이더 (5 ~ 60)
- 빠른 프리셋 4개: **기본 수렴 / 느린 수렴 / 빠른 수렴 / 💥 발산**
- Loss Landscape + 이동 경로 PNG 저장 (`output/gd_path.png`)
- 학습률 4종 비교 PNG 저장 (`output/lr_comparison.png`)
- 전체 Step 상세 테이블 (x, f(x), ∇f(x))

### 🔬 Tab 2 — Hooke's Law TensorFlow 학습
- TensorFlow `Dense(1)` 선형 회귀로 스프링 상수 `k = 10 N/m` 학습
- Epoch별 Loss 곡선 PNG 저장 (`output/loss_curve.png`)
- 회귀 적합선 PNG 저장 (`output/regression_fit.png`)
- 가중치 수렴 PNG 저장 (`output/weight_convergence.png`)
- 질량 입력 → 스프링 변위 예측 + SVG 스프링 애니메이션
- 예측 결과 PNG 저장 (`output/prediction_result.png`)

### 🖼️ Tab 3 — 결과 갤러리
- 생성된 PNG 파일 자동 표시 및 라이트박스 확대

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| Gradient Descent | 기울기 반대 방향으로 이동해 최솟값 탐색 |
| Learning Rate | 한 번에 이동하는 보폭, 너무 크면 발산 |
| 수렴 조건 | `\|x\| < 0.05` 이면 최솟값 도달로 판정 |
| 발산 조건 | 학습률 ≥ 1.0 이면 발산 위험 |
| Hooke's Law | `F = k·x`, `x = mg/k` — 스프링 탄성력 |
| Linear Regression | `y = Wx + b` — TF가 W(≈g/k)를 경사 하강으로 학습 |

---

## 정확도 검증 결과

| 항목 | 결과 |
|------|------|
| 훈련 정확도 | **99.88%** (목표 98% 초과) |
| R² Score | **0.9988** |
| 예측 정확도 (2.5 kg) | **99.96%** → 245.11 cm (이론: 245.0 cm) |
| k 추정값 | **10.004 N/m** (실제: 10 N/m) |

---

## 출력 파일 (`output/`)

| 파일 | 설명 |
|------|------|
| `gd_path.png` | 경사 하강 이동 경로 (Loss Landscape + Step별 Loss) |
| `lr_comparison.png` | 학습률 4종 비교 (Slow / Optimal / Fast / Diverging) |
| `loss_curve.png` | TF 훈련 Epoch별 MSE Loss 곡선 |
| `regression_fit.png` | 훅의 법칙 회귀 적합선 + 잔차 |
| `weight_convergence.png` | 가중치(W) · 편향(b) 수렴 과정 |
| `prediction_result.png` | 스프링 다이어그램 + 예측 vs 이론값 비교 |

---

## 프로젝트 구조

```
week2/Gre_Des_Vis/
├── main.py              # FastAPI 백엔드
├── gd_vis.py            # 경사 하강법 시각화 (NumPy + Matplotlib)
├── hooke_model.py       # TensorFlow 훅의 법칙 선형 회귀
├── requirements.txt
├── templates/
│   └── index.html       # Tailwind CSS 전문가 수준 UI
└── output/              # PNG 자동 저장
```

---

## 서버 실행

```bash
cd week2/Gre_Des_Vis
pip install -r requirements.txt
python main.py
# 또는
uvicorn main:app --reload --port 8003
```

→ **http://localhost:8003**

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 백엔드 | FastAPI + Uvicorn |
| 머신러닝 | TensorFlow 2.x (Keras) |
| 시각화 | Matplotlib (PNG 출력) |
| 수치 연산 | NumPy |
| 프론트엔드 | Tailwind CSS (CDN) + Vanilla JS |
| 폰트 | Inter + JetBrains Mono |
