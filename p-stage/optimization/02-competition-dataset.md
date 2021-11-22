# 대회 및 데이터셋 소개

### Table of Contents

1. 대회 소개 & FLOPs
2. Dataset 및 task 소개
3. EDA 및 Dataset 커스터마이징 과정



## 1) 대회 소개 & FLOPs

### 1.1) 대회 목표

경량화를 평가하기? Inference 속도를 대회 기준으로 설정. Image classification 경량화에 집중.

### 1.2) 대회 평가 기준

성능이 좋으면서 inference 속도가 가장 빠른 모델 찾기

 ![Screen Shot 2021-11-22 at 9.29.11 PM](boostcamp-ai-tech/assets/images/Screen Shot 2021-11-22 at 9.29.11 PM.png)

### 1.3) FLOPs에 대해서

> 플롭스(**FLOPS**, FLoating point Operations Per Second)는 컴퓨터의 성능을 수치로 나타낼 때 주로 사용되는 단위이다. 초당 부동소수점 연산이라는 의미로 컴퓨터가 1초동안 수행할 수 있는 부동소수점 연산의 횟수를 기준으로 삼는다.

연산 횟수는 속도에 영향을 주는 **간접적인 요소** 중 하나.

여러 요소가 복합적으로 작용하기 때문에 간접적 요소.

**ShuffleNetv2**: 속도에 영향을 주는 요소에 대한 insight를 얻을 수 있는 논문.



## 2) Dataset 및 Task 소개

### 2.1) TACO

- Trash Annotations in Context Dataset

### 2.2) 대회용 데이터셋 소개

- Object detection task를 위해 제작된 데이터셋. 



## 3) EDA 및 Dataset 커스터마이징 과정

### 3.1) 간단 EDA

총 6개의 Category

- train + valid : 20851
- test: 5217
  - private: 2611
  - public: 2606



