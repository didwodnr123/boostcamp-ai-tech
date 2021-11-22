# Intro to Optimization



## 1.1 경량화의 목적

### On-device limitation

- 배터리 소모량
- RAM 메모리 사용량
- 저장공간
- 컴퓨팅 파워

### AI on cloud

- Latency (한 요청의 소요 시간, 단위 시간당 처리 가능한 요청 수)
- 리소스 사용 비용

### Computation as a key component of AI Progress

- Model이 커짐에 따라 연산량이 기하급수적으로 증가함



## 1.2 경량화 분야

### 네트워크 구조 관점

- Efficient Architecture Design (경량 모델을 디자인 하는것)
- Network Pruning (중요도가 낮은 파라미터를 제거하는 것)
- Knowledge Distillation
- Matrix/Tensor Decomposition (연산을 더 작은 구조?로 낮추는 것)

### Hardware 관점

- Network Quantization (float32 -> int)
- Network Compiling (inference를 효과적으로 하기 위한 네트워크를 컴파일)



#### Efficient architecture design, 모델 찾기

- Software 1.0: 사람이 짜는 모듈
- Software 2.0: 알고리즘이 찾는 모듈
  - 사람의 직관을 상회하는 성능의 모듈들을 찾아낼 수 있음 (AutoML)

#### Network Pruning, 찾은 모델 줄이기

- 중요도가 낮은 파라미터를 제거 (중요도를 찾는 것 포함)

#### Knowledge distillation

- 학습된 큰 네트워크를 작은 네트워크의 학습 보조로 사용하는 방법 (Teacher network & Student network)

#### Matrix/Tensor decomposition

- 하나의 Tensor를 작은 Tensor들의 operation들의 조합(합, 곱)으로 표현하는 것

#### Network Quantization

- float32 -> float16 or int8 ...

#### Network Compiling

- 사실상 **속도에 가장 큰 영향**을 미치는 기법
- 몇 개의 레이어를 하나로 묶는 방법? 
- 어떤 레이어를 어떻게 묶냐에 따라 성능차이가 발생
- 어떤 GPU를 쓰느냐에 따라도 다름



## 2.1 강의 목표

- 경량화, 최적화는 매우 넓은 분야
- 가능한 broad한 토픽으로 구성

**AutoML을 위주로 학습할 예정.** 

**Matrix/Tensor Decomposition, Network Quantization, Network Compiling을 다룰 예정.**

> Network Pruning or Knowledge distillation은 활발히 연구되고 있지만, 일반화하여 적용하기는 아직 어려움.

#### 일련의 과정

> 경량 모델 탐색(AutoML) -> 찾은 모델 쪼개기(Tensor Decomposition) -> 쪼갠 모델 꾸겨넣기(Quantization & Compile)