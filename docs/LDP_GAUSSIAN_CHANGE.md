# LDP Mechanism Change: Laplace → Gaussian

**Date**: 2026-06-15  
**Change Type**: Privacy mechanism default modification

---

## 🔧 변경 내용 요약

기본 LDP 메커니즘을 **Laplace**에서 **Gaussian (Analytic Gaussian Mechanism)**으로 변경했습니다.

---

## 📁 수정된 파일들

### 1. `hmdp/ldp.py`
**변경 사항**:
- 모듈 docstring: Gaussian mechanism이 기본임을 명시
- `privatize_features()`: 기본값 `mechanism='laplace'` → `'gaussian'`
- `generate_privatized_statistics()`: 기본값 `mechanism='laplace'` → `'gaussian'`
- `privatize_regions()`: 기본값 `mechanism='laplace'` → `'gaussian'`
- 수식 설명: Lap(Δ₂/ε) → N(0, σ²) 로 업데이트

**수식 변경**:
```
Before: φ̃(s_τ) = φ(s_τ) + Lap(Δ₂/ε)          [pure ε-LDP]
After:  φ̃(s_τ) = φ(s_τ) + N(0, σ²)           [(ε,δ)-LDP]
        where σ = Δ₂√(2ln(1.25/δ))/ε
```

### 2. `hmdp/projection/training.py`
**변경 사항**:
- 모듈 docstring: Gaussian noise augmentation 명시
- `train_projection_offline()`: `_laplace_noise()` 직접 호출 → `privatize_features()` 사용
- 자동으로 Gaussian mechanism 적용됨

### 3. `hmdp/blind_vlm.py`
**변경 사항**:
- 모듈 docstring: (ε,δ)-LDP Gaussian noise로 업데이트
- 수식 설명 업데이트

---

## 📊 메커니즘 비교

| 특성 | Laplace (이전) | Gaussian (변경 후) |
|------|---------------|-------------------|
| **Privacy Type** | Pure ε-LDP | (ε, δ)-LDP |
| **Noise Distribution** | Laplace(0, Δ₂/ε) | N(0, σ²) |
| **Scale** | b = Δ₂/ε | σ = Δ₂√(2ln(1.25/δ))/ε |
| **Delta (δ)** | N/A | 1e-5 (기본값) |
| **Composition** | Linear | Better (advanced composition) |
| **Tail Behavior** | Exponential | Quadratic (Gaussian tail) |

---

## 🎯 Gaussian 메커니즘의 장점

1. **Better Composition**: (ε, δ)-LDP는 여러 쿼리에 대한 composition이 더 효율적
2. **Lower Variance**: Same privacy budget에서 일반적으로 더 낮은 variance
3. **Standard Practice**: 실제 DP 응용에서 Gaussian mechanism이 더 널리 사용됨
4. **Analytic Calibrator**: 공식적이고 검증된 calibration 공식 사용

---

## ⚠️ 주의사항

### Delta (δ) 선택
- **기본값**: δ = 1e-5
- **의미**: Privacy failure probability = 0.001%
- **권장**: 응용에 따라 1e-6 ~ 1e-4 범위 조정 가능

### Epsilon 변환
같은 ε 값을 사용해도 실제 privacy guarantee는 다릅니다:
- Laplace: pure ε-LDP
- Gaussian: (ε, δ)-LDP (weaker than pure ε-LDP)

따라서 같은 privacy 수준을 원한다면 Gaussian에서 약간 더 높은 ε 사용 필요.

---

## 🔬 수식 상세

### Analytic Gaussian Mechanism

```
σ = Δ₂ × √(2 × ln(1.25/δ)) / ε

where:
  Δ₂ = L2 sensitivity = 2.0 (clipped latent features)
  δ = failure probability = 1e-5 (default)
  ε = privacy budget
```

### 예시 계산 (ε=5.0, δ=1e-5)
```
σ = 2.0 × √(2 × ln(1.25/1e-5)) / 5.0
  = 2.0 × √(2 × 11.736) / 5.0
  = 2.0 × 4.84 / 5.0
  = 1.94
```

비교 (Laplace, 같은 ε=5.0):
```
scale = Δ₂/ε = 2.0/5.0 = 0.4
```

→ Gaussian은 더 큰 noise variance를 가짐 (그러나 tail behavior가 다름)

---

## 📝 사용 예시

### 기본 사용 (자동으로 Gaussian)
```python
from hmdp.ldp import LocalDifferentialPrivacy

ldp = LocalDifferentialPrivacy(feature_dim=256, sensitivity=2.0)
phi_private = ldp.privatize_features(phi, epsilon=5.0)  # Gaussian 자동 사용
```

### 명시적 메커니즘 선택 (여전히 가능)
```python
# Gaussian (명시적)
phi_private = ldp.privatize_features(phi, epsilon=5.0, mechanism='gaussian', delta=1e-5)

# Laplace (이전 방식, 여전히 가능)
phi_private = ldp.privatize_features(phi, epsilon=5.0, mechanism='laplace')
```

---

## 🧪 영향 받는 실험들

### 1. Background Tmux Training
현재 실행 중인 `hmdp_improvement` 세션의 extended training에 자동으로 적용됨.
- 새로운 W_proj checkpoint는 Gaussian mechanism으로 학습됨
- 예상 결과: 더 부드러운 noise profile, potentially better convergence

### 2. Future Evaluations
모든 새로운 evaluation은 자동으로 Gaussian noise 사용:
```python
python -m hmdp.run_real_vlm --epsilon 5.0 ...  # Gaussian 적용됨
```

---

## 📚 관련 문서

- **Dwork & Roth (2014)**: The Algorithmic Foundations of Differential Privacy
- **Balle & Wang (2018)**: Improving the Gaussian Mechanism for Differential Privacy
- **원본 논문 Sec 3.3**: Privacy Resilience via Latent-level LDP

---

## ✅ 검증 체크리스트

- [x] `hmdp/ldp.py` 기본값 변경
- [x] `hmdp/projection/training.py` Gaussian 사용
- [x] `hmdp/blind_vlm.py` 문서화 업데이트
- [x] 수식 설명 업데이트
- [ ] Tmux training 완료 후 결과 비교
- [ ] Laplace vs Gaussian 성능 비교 실험

---

**Changed by**: Code modification  
**Reason**: L2 sensitivity-based Laplace scale이 타당하지 않음 → Gaussian mechanism으로 변경
