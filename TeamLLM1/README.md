# 법률 QA 챗봇 프로젝트 ⚖️

이 프로젝트는 부동산법 및 근로법에 대한 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

---

## 📂 폴더 구조

```
.
├── data/
│   ├── raw/
│   └── vector_db/
├── evaluation/
│   ├── metrics.py
│   └── pipeline.py
├── models/
│   ├── base_model/
│   ├── service_model/
│   └── setup_model.py
├── service/
│   ├── interface/
│   ├── langgraph/
│   └── main.py
├── training/
│   ├── eval.py
│   ├── metrics.py
│   └── train.py
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 프로젝트 설정 및 실행 가이드

### 1. 환경 설정

```bash
# (권장) 가상 환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# .env 파일 설정: .env.example을 복사하여 .env 파일을 만들고,
# HUGGINGFACE_HUB_TOKEN, 모델 이름, 경로 등을 설정합니다.
cp .env.example .env
```

### 2. 의존성 라이브러리 및 패키지 설치

`requirements.txt`에 명시된 모든 라이브러리를 설치하고, 프로젝트를 패키지 형태로 설치하여 어디서든 모듈을 쉽게 불러올 수 있도록 설정합니다.

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. 기반 모델 다운로드

`.env` 파일에 설정된 LLM과 임베딩 모델을 `models/base_model/` 경로에 다운로드합니다. 이 명령어는 `setup.py`에 정의된 `download-models` 커맨드를 실행합니다.

```bash
download-models
```

### 4. FAISS 벡터 인덱스 생성

RAG의 핵심인 검색 데이터베이스를 구축하는 과정입니다. `data/vector_db/data_preprocessing.py` 스크립트는 원본 데이터를 임베딩하여 FAISS 인덱스를 생성하고 저장합니다.

```bash
python data/vector_db/data_preprocessing.py
```

> 여기까지 작업 완료...
