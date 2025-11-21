# Dinov3 기반 반도체 AI Agent Tool

이 레포지토리는 **Dinov3 (Vision Transformer)** 를 기반으로 반도체 도메인(Semiconductor Domain)에서 활용 가능한 **AI Agent Tool**을 연구 및 개발하기 위한 프로젝트입니다.

주로 반도체 제조 공정 중 발생할 수 있는 결함(Crack 등)을 탐지하고, 이미지를 분석하여 의미 있는 특징(Feature)을 추출하는 다양한 실험적 코드와 도구들을 포함하고 있습니다.

## 📌 프로젝트 개요

- **목표**: 반도체 이미지 데이터에 특화된 비전 AI 에이전트 개발
- **핵심 기술**: 
  - **Dinov3**: Self-Supervised Learning 기반의 강력한 Vision Backbone 사용
  - **Unsupervised Segmentation**: 레이블 없이도 이미지 내의 이질적인 영역(결함 등)을 군집화(Clustering)하여 탐지
  - **Feature Engineering**: Sobel 필터 등을 활용한 Edge Feature와 Deep Learning Feature의 결합

## 📂 폴더 구조 (Directory Structure)

주요 폴더 및 파일에 대한 설명입니다.

```
c:\workspace\dinov3
├── dinov3/                  # Dinov3 코어 라이브러리 및 모델 정의
├── my_codes/                # 주요 기능 구현 스크립트 모음
│   ├── extract_dino_features.py  # DINO 특징 추출 스크립트
│   ├── finetune_ddp.py           # 분산 학습(DDP) 파인튜닝 스크립트
│   ├── generate_masks.py         # 마스크 생성 도구
│   └── ...
├── weights/                 # 학습된 모델 가중치 저장소
├── imgs/                    # 입력 이미지 데이터 (예: crack_hardcases)
├── output/                  # 모델 추론 결과 및 시각화 저장소
├── my_semiconductor_config.yaml # 반도체 데이터셋 학습 설정 파일
├── requirements.txt         # 의존성 패키지 목록
└── README.md                # 프로젝트 설명 파일
```

## ✨ 주요 기능 (Key Features)

### 1. 결함 탐지 및 분할 (Defect Detection & Segmentation)
- 현재 주요 실험이 진행 중인 스크립트입니다.
  - DINOv3의 Feature와 이미지의 Edge Feature(대각선/가로선)를 결합합니다.
  - K-Means Clustering을 통해 배경과 이질적인 결함(Crack) 영역을 자동으로 탐지합니다.
  - PCA를 이용한 Feature Map 시각화 및 결과 오버레이 이미지를 생성합니다.

### 2. 모델 파인튜닝 (Fine-tuning)
- **`my_semiconductor_config.yaml`**: 반도체 데이터셋(ADE20K 포맷 등)에 맞춰 모델을 학습시키기 위한 설정 파일입니다.
- **`finetune.py` / `my_codes/finetune_ddp.py`**: 설정된 Config를 바탕으로 모델을 미세 조정(Fine-tuning)합니다.

### 3. 특징 추출 및 분석 (Feature Extraction)
- 이미지에서 고차원 특징을 추출하여 분석하거나, 유사도 기반의 검색/분류 작업에 활용할 수 있는 도구들을 제공합니다.

