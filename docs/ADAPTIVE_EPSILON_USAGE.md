# Adaptive Per-Region Epsilon Governor Guide

SAC meta-policy를 real-VLM H-MDP 파이프라인에 연결해, 균일 프라이버시 예산 broadcast `[epsilon] * M`을 **region별 적응형 예산 벡터**로 대체하는 기능 사용 가이드.

## 개요

`run_real_vlm.py`의 기본 동작은 모든 region에 동일한 `--epsilon`을 적용하는 균일 LDP였습니다.
`--adaptive` 플래그를 켜면, 복원된 SAC meta-policy가 거버넌스 상태 `s_t`로부터
region별 ε 벡터(와 GoT의 추론 폭 `k`)를 선택합니다.

```
s_t = [ U_t , λ_t^(1), ..., λ_t^(M) ]      (1 + M = 26 차원, M = 25)
```

- `U_t` : 스칼라 추론 불확실성 feature. 스텝별 `u_t`는 GoT aggregation **이후**에만
  알 수 있는 반면, 프라이버시 예산은 privatization **이전**에 정해야 합니다.
  따라서 governor는 carry된 값을 사용합니다 — 첫 상태에서는 `U_0`(`u_init`),
  이후에는 (선택적으로) 직전 스텝의 `u_t`.
- `λ_t` : privatization **이전**의 clean proxy latent `φ`에 대한 region별 latent
  Shannon entropy. `ExecutionEngine.compute_latent_entropy` (Eq. 1)로 계산.

연속 SAC action `a ∈ [-1, 1]^26`은 `HMDPConfig.action_to_params`로
`(epsilons: List[float] 길이 M, k: int)`에 매핑되며, 이 함수가 ε을
`[epsilon_min, epsilon_max]`, `k`를 `[k_min, k_max]`로 clip합니다.

## Quick Start

### 1. 균일 ε (기존 동작, 기본값)

```bash
python3 -m hmdp.run_real_vlm \
    --models qwen2.5-vl-7b \
    --num-samples 100 \
    --epsilon 5.0 --k 5 \
    --wproj-ckpt checkpoints/wproj_eq8.pt
```

### 2. 적응형 per-region ε (체크포인트 필요)

```bash
python3 -m hmdp.run_real_vlm \
    --models qwen2.5-vl-7b \
    --num-samples 100 \
    --wproj-ckpt checkpoints/wproj_eq8.pt \
    --adaptive \
    --sac-ckpt /path/to/sac_meta_policy.pt
```

체크포인트가 로드 가능하면 다음과 같이 출력됩니다:

```
  [SAC-governor] adaptive SAC governor active (ckpt=/path/to/sac_meta_policy.pt)
  Epsilon policy: ADAPTIVE per-region
```

## CLI 플래그

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--adaptive` | flag | off | SAC meta-policy로 per-region ε(와 k) 선택. 끄면 균일 `--epsilon` broadcast. |
| `--sac-ckpt` | str | None | 학습된 SAC meta-policy 체크포인트 경로 (`SACMetaPolicy.save` 포맷). 추론에는 `actor` 서브-스테이트만 필요. |
| `--sac-carry-uncertainty` | flag | off | sample 간 `u_t`를 다음 상태의 `U_t`로 carry (시뮬레이션 multi-step 의미론). 기본값은 sample별 reset (순서 독립, 재현 가능). |

`--adaptive`를 켜되 `--sac-ckpt`를 주지 않거나, 체크포인트가 로드 불가능하면
**경고를 출력한 뒤 균일 ε으로 graceful degradation** 합니다 (아래 참조).

주의: 현재 로컬 `checkpoints/hmdp/*.pt` 파일은 `torch.load` 검증 기준으로
손상되어 있으므로 adaptive 결과 보고에 사용하면 안 됩니다. 새로 학습했거나
로드 검증을 통과한 SAC actor checkpoint를 `--sac-ckpt`로 지정하세요.

## 동작 방식

`--adaptive`가 활성화되고 체크포인트가 로드되면, 두 평가 경로 모두에서:

1. privatization **이전**에 clean proxy latent `φ` (M, d)를 얻습니다.
   - **Blind-but-Smart 경로** (기본): `pipe.dino.extract_regions(img)` →
     `pipe.proxy_encoder(...)`로 clean `φ`를 별도 추출한 뒤 `pipe.ldp.privatize_regions`로 privatize.
   - **raw-pixel 경로** (`--raw-pixel` ablation): 이미 산출된 `phi_grid`를 사용.
2. `governor.compute_epsilons(phi_clean)` → `λ_t = entropy(φ)`로 `s_t`를 구성하고
   SAC actor가 `(epsilons_t, k_t)`를 반환.
3. `epsilons_t`로 region별 privatize, `k_t`로 GoT k-path 추론.
4. GoT aggregation 후 얻은 `u_t`를 `governor.update_uncertainty(u_t)`로
   다음 상태의 `U_t`에 carry (carry 정책은 `--sac-carry-uncertainty`에 따름).

`reset_episode()`는 sample 루프 시작 전에 한 번 호출되어 `U_t`를 `u_init`으로 초기화합니다.

## 패키지 레이아웃

governor는 전체 H-MDP 시뮬레이션 모듈(`config`, `sac_policy`, `execution_engine`)을
**`hmdp_sim`** 패키지로 import합니다. 다른 위치에 설치된 경우 `HMDP_SIM_PKG`
환경변수로 패키지명을 override합니다.

```
project_root/
├── hmdp/                    # real-VLM 트리
│   ├── run_real_vlm.py      # 평가 엔트리포인트 (--adaptive 통합)
│   ├── sac_governor.py      # AdaptiveEpsilonGovernor (NEW)
│   ├── blind_vlm.py
│   ├── ldp.py
│   ├── ltm.py
│   └── ...
└── hmdp_sim/                # 시뮬레이션 패키지 (config / sac_policy / execution_engine)
    ├── __init__.py
    ├── config.py            # HMDPConfig, action_to_params
    ├── sac_policy.py        # SACMetaPolicy
    └── execution_engine.py  # ExecutionEngine.compute_latent_entropy (Eq. 1)
```

패키지명 해석 순서: `$HMDP_SIM_PKG` → `"hmdp_sim"`.

```bash
# 시뮬레이션 패키지가 다른 이름으로 설치된 경우
export HMDP_SIM_PKG=my_hmdp_package
python3 -m hmdp.run_real_vlm --adaptive --sac-ckpt ckpt.pt ...
```

해당 패키지의 `ExecutionEngine`이 `compute_latent_entropy`를 노출하지 않으면
(예: projection-only로 trimmed된 변형) import가 거부되어, 잘못된 심볼을
조용히 집어드는 것을 방지합니다.

## Graceful Degradation

다음 경우 governor는 스스로 비활성화되고 `compute_epsilons`는
`fallback_epsilon`/`fallback_k`로 구성한 **균일 벡터**(= 기존 동작)를 반환합니다.
모두 **명시적 경고**를 출력하며, 조용히 넘어가지 않습니다.

- 시뮬레이션 패키지 import 실패
- 체크포인트 미지정 (`--sac-ckpt` 없음)
- 체크포인트 파일 없음
- `torch.load` 실패 (예: truncate/corrupt된 `.pt`, zip central directory 부재)
- 체크포인트에서 actor 서브-스테이트를 찾을 수 없음
- 필수 actor layer(`backbone.0.weight`, `mean_head.weight`) 누락
  → policy가 사실상 무작위이므로 의미 없는 "adaptive" 실행을 광고하지 않도록 실패 처리

추론에는 actor 서브-스테이트만 필요하며, critic이 없는 부분(partial) 체크포인트도 허용됩니다.

## 재현성 노트

`U_t` carry 정책은 두 가지 모드가 있습니다 (`reset_per_sample`, CLI `--sac-carry-uncertainty`로 제어):

- **`reset_per_sample=True` (기본, `--sac-carry-uncertainty` off)**: 매 sample 시작 시
  `U_t`를 `u_init`으로 reset. 선택된 예산은 현재 sample의 `λ_t`(와 고정 `U_0`)에만
  의존하므로 **sample 평가 순서와 무관하고 재현 가능**합니다. 이 모드에서
  `update_uncertainty`는 (bookkeeping 외) no-op입니다.
- **`reset_per_sample=False` (`--sac-carry-uncertainty` on)**: 직전 스텝의 `u_t`를
  연속 호출에 걸쳐 carry (시뮬레이션 multi-step carry와 일치). 이 모드에서는
  결과가 **순서 의존적**이므로 재현을 위해 고정된 평가 순서가 필요합니다.

real-VLM 평가의 sample들은 서로 독립이므로 기본값은 per-sample reset입니다.

## Python API

```python
from hmdp.sac_governor import AdaptiveEpsilonGovernor

gov = AdaptiveEpsilonGovernor(
    ckpt_path="sac_meta_policy.pt",   # None/없음/손상 시 균일 fallback
    device="cpu",                     # "cuda" / "cuda:0" / "mps" / "cpu" (자동 fallback)
    num_regions=25,
    u_init=0.5,
    fallback_epsilon=5.0,             # 비활성화 시 사용할 균일 ε
    fallback_k=5,
    deterministic=True,               # SAC action을 deterministic하게 선택
    reset_per_sample=True,            # U_t carry 정책 (재현성 노트 참조)
    verbose=True,
)

gov.reset_episode()                              # sample 시작
epsilons, k = gov.compute_epsilons(phi_clean)    # phi_clean: (M, d) clean latent
# ... privatize_regions(phi_clean, epsilons) → GoT(k) → u_t 획득 ...
gov.update_uncertainty(u_t)                      # 다음 상태 U_t로 carry

gov.enabled    # True면 adaptive, False면 균일 fallback
```

### 주요 메서드

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `reset_episode()` | None | carry된 `U_t`를 `u_init`으로 초기화. sample 시작 시 호출. |
| `update_uncertainty(u_t)` | None | post-GoT `u_t`를 다음 상태 `U_t`로 carry. `reset_per_sample=True`면 no-op. |
| `build_state(phi_clean)` | Tensor `(1+M,)` | `s_t = [U_t ; λ_t^(1..M)]` 구성. `(B,M,d)` 입력은 batch 평균. |
| `compute_epsilons(phi_clean)` | `(List[float] 길이 M, int)` | adaptive ε 벡터와 k. 비활성화 시 균일 fallback. |

## 테스트

`test_adaptive_epsilon.py`는 체크포인트 없이도 실행 가능한 smoke-test입니다
(시뮬레이션 패키지를 `hmdp_sim`으로 import 가능해야 함):

```bash
# 시뮬레이션 패키지 위치 지정 (필요 시)
export HMDP_SIM_PKG=hmdp_sim
python3 test_adaptive_epsilon.py
```

검증 항목:

1. 체크포인트 미지정 → 균일 fallback (`enabled=False`)
2. 손상된 체크포인트 → graceful fallback
3. `build_state` 출력 shape `(26,)`
4. `reset_per_sample` 두 regime 동작 (no-op vs carry)
5. device 문자열 fallback (`cuda:0` → CPU)

## 주의사항

1. **체크포인트 무결성**: 다른 서버에서 다운로드한 `.pt`가 truncate되면
   (zip central directory 부재) `torch.load`가 실패하고 균일 fallback으로 동작합니다.
   전송 후 sha256 검증을 권장합니다.
2. **device**: SAC actor는 작아 CPU로 충분합니다. `mps`/`cuda` 미가용 시 자동으로 CPU로 fallback합니다.
3. **clean φ 노출 경계**: governor는 `λ_t` 계산을 위해 privatization **이전**의
   clean `φ`에 접근합니다. 이는 거버넌스(예산 선택) 용도이며 VLM에는 privatize된
   latent만 주입됩니다 — Blind-but-Smart 경계는 유지됩니다.

## 파일 구조

```
hmdp/
├── run_real_vlm.py           # 평가 엔트리포인트 (--adaptive 통합)
├── sac_governor.py           # AdaptiveEpsilonGovernor (NEW)
test_adaptive_epsilon.py      # smoke-test
ADAPTIVE_EPSILON_USAGE.md     # 이 문서
```
