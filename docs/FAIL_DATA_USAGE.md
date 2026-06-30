# Fail Data Integration Guide

GUI-360 Fail 데이터를 RL 학습 파이프라인에 통합하는 방법

## 개요

- **Fail 데이터**: 1,093,525개 샘플 (reward: 0.0 ~ 0.5, 평균 0.47)
- **Success 데이터**: 101,800개 샘플 (reward = 1.0)
- **용도**: SAC RL 학습 시 negative 샘플 (실패 상황 학습)

## 설정 방법

### 1. Config 수정 (`hmdp/config.py`)

```python
config = HMDPConfig()
config.gui360_fail_data_path = "/path/to/gui360_full/converted_fail_data.json"
config.fail_data_ratio = 0.3  # 30% 비율로 fail 샘플 혼합
```

### 2. Environment 생성

```python
from hmdp.gui_env import GUIEnvironment

env = GUIEnvironment(
    data_path=config.gui360_data_path,      # success 데이터
    image_base_path=config.gui360_image_base,
    fail_data_path=config.gui360_fail_data_path,  # fail 데이터
    fail_ratio=0.3,  # 0.0 = success만, 1.0 = fail만
)

# 통계 확인
print(env.get_stats())
# {
#   'success_samples': 101800,
#   'fail_samples': 1093525,
#   'fail_ratio': 0.3,
#   'fail_reward_mean': 0.47,
#   'fail_reward_std': 0.12
# }
```

### 3. 학습 루프에서 사용

```python
for episode in range(num_episodes):
    state = env.reset()  # 자동으로 success/fail 혼합 샘플링
    
    # is_fail_sample 체크 가능
    if state.is_fail_sample:
        print(f"Episode {episode}: [FAIL] sample (reward={state.true_reward})")
    
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(...)
        
        # SAC replay buffer에 저장
        # - success: reward = 1.0 (성공) 또는 0.0 (실패)
        # - fail: reward = true_reward * (0.5 + 0.5 * IoU)
        
        agent.store_transition(state, action, reward, next_state, done)
```

## Fail 샘플링 메커니즘

### 가중치 샘플링

Fail 데이터는 **reward 값에 비례**하여 샘플링됨:

```python
# 높은 reward (0.5에 가까운) 실패 케이스가 더 자주 샘플링
weights = fail_rewards + 0.1  # stability constant
probs = weights / weights.sum()
```

이유: 완전 실패(0.0)보다 "거의 성공"한 케이스(0.5)가 학습에 더 유용함

### Reward 계산

| 데이터 타입 | Reward 계산 방식 |
|------------|-----------------|
| **Success** | `1.0` if correct, else `0.0` (binary) |
| **Fail** | `true_reward * (0.5 + 0.5 * IoU)` (scaled) |

## 활용 시나리오

### 1. Offline RL (CQL/IQL)

Fail 데이터를 static dataset에 포함하여 conservative Q-learning:

```python
# Mixed dataset for offline RL
dataset = success_data + random.sample(fail_data, k=int(len(success_data)*0.3))
```

### 2. Contrastive Learning

성공/실패 쌍을 활용한 대조 학습:

```python
# Success embedding vs Fail embedding 분리
success_emb = encoder(success_sample)
fail_emb = encoder(fail_sample)

contrastive_loss = margin_ranking_loss(success_emb, fail_emb, margin=1.0)
```

### 3. H-MDP SAC 학습

Meta-policy가 fail 상황에서도 적절한 (ε, k) 선택 학습:

```python
# Fail episode에서도 보상 기반 업데이트
if info["is_fail_sample"]:
    # 더 aggressive한 exploration (높은 k)
    meta_reward = compute_pes_reward(success=False, epsilon=eps, k=k)
```

## 테스트

### Python API 테스트

```python
from hmdp.gui_env import GUIEnvironment
import numpy as np

# Fail 데이터 환경 생성
env = GUIEnvironment(
    data_path="/path/to/gui360_data.json",
    image_base_path="/path/to/images",
    fail_data_path="/path/to/converted_fail_data.json",
    fail_ratio=0.3,  # 30% fail 샘플링
)

# 통계 확인
stats = env.get_stats()
print(f"Success samples: {stats['success_samples']}")
print(f"Fail samples: {stats['fail_samples']}")
print(f"Fail ratio: {stats['fail_ratio']}")

# 에피소드 테스트
for episode in range(10):
    state = env.reset()
    print(f"Episode {episode}: {'[FAIL]' if state.is_fail_sample else '[SUCCESS]'} "
          f"reward={state.true_reward:.2f}")
```

## 주의사항

1. **메모리**: fail 데이터는 1.5GB JSON 파일. lazy loading 고려 필요
2. **속도**: 109만 개 샘플 중 샘플링 시 인덱싱 최적화 필요 (현재는 numpy array 사용)
3. **밸런스**: fail_ratio가 너무 높으면 (0.5+) agent가 pessimistic해질 수 있음
4. **Reward 스케일**: success (0/1) vs fail (0~0.5) 스케일 차이 주의

## 확장 아이디어

- **Curriculum Learning**: 초기엔 success만 → 점진적으로 fail 증가
- **Hard Negative Mining**: 특정 유형의 fail 케이스 오버샘플링
- **Trajectory-level**: 현재는 single-step, multi-step fail trajectory 지원 확장
