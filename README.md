# 🧑🏻‍🦳 노인들의 쇼핑 편리성을 높이는 STT 장바구니 서비스

> **2025 XAI-ADV Session MM Team**\
> *“그 왜 파란색 봉지 과자 두어 개 담아줘봐라”*

---


## 🧠 프로젝트 개요

- **프로젝트명**: 노인들의 쇼핑 편리성을 높이는 STT 장바구니 서비스
- **목표**: STT 및 Text2Image Retrieval을 통해 단순 발화로도 사용 가능한 음성 기반 장바구니 서비스 구현
- **핵심 과제**:
  - Whisper를 통한 사용자 입력 음성 처리
  - 한국어 특화 Koclip 파인튜닝을 통한 모달리티 간 정렬이 이루어진 임베딩 모델 학습
  - Text2Image Retrieval 로직 구현

---

## 🧩 프로젝트 구조

```bash
📁 STT-Shopping-Cart/
├── TTS/                  
│   └── TTS_module.py
├── Koclip/
│   ├── Koclip_finetuning_dataset.py      #Kanana-1.5-v-3B-Instruct 모델 활용 이미지 캡셔닝 및 메타데이터 결합
│   ├── Koclip_finetuning_dataset_cleansing.py      #AI-Hub 상품이미지 데이터셋 처리
│   └── Koclip_finetuning.py      #Training
├── Text2image Retireval/
│   ├── Vectordb_upsert.py      #Pinecone VectorDB 구성
│   └── Search_item.py      #벡터DB내 상품 검색
├── UI/
│   └── STT_to_Gradio.py
└── README.md
