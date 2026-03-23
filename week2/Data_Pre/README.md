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
