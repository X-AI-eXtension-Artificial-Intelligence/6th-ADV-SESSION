# 법률 QA 챗봇 프로젝트 ⚖️

이 프로젝트는 부동산법 및 근로법에 대한 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

---

## 📂 폴더 구조

```
.
├── conversations/
├── data/
│   ├── raw/
│   │   ├── df.csv
│   │   ├── new_law.csv
│   │   └── chunking.py
│   └── vector_db/
│       └── setup_db.py
├── evaluation/
│   ├── metrics.py
│   └── pipeline.py
|── legal_chatbot
│   └── cli.py
├── models/
│   ├── rerank/
│   ├── retrieve/
│   └── setup_model.py
├── service/
│   ├── interface/    # Streamlit UI 코드드
│   ├── langgraph/    # LangGraph API 코드
│   └── main.py       # Streamlit, API 띄우기
├── training/
│   ├── eval.py
│   ├── metrics.py
│   └── train.py
|── .env.example
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 프로젝트 설정 및 실행 가이드

### 1. 가상환경 설정
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. 환경 변수 설정
.env 파일 설정: .env.example을 복사하여 .env 파일을 만들고,
HUGGINGFACE_HUB_TOKEN, 모델 이름, 경로 등을 설정합니다.
```bash
# Windows PowerShell
copy .env.example .env
# macOS / Linux
cp .env.example .env
```


### 3. 의존성 라이브러리 및 패키지 설치

`requirements.txt`에 명시된 모든 라이브러리를 설치하고, 프로젝트를 패키지 형태로 설치하여 어디서든 모듈을 쉽게 불러올 수 있도록 설정합니다.

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. 모델 및 데이터 초기화

`.env`에 지정된 모델과 데이터를 다운로드하고 전처리합니다.

```bash
setup-all
```

### 5. Fast API 실행
LangGraph에 query문을 전송하기 위해 API Router를 실행합니다.

```bash
python -m service.main
```

해당 코드를 실행하면 기본적으로 http://localhost:20000에서 API가 실행됩니다.

### 6. Streamlit 코드 실행
service/inference/gradio_ui.py 이름 Streamlit.py 수정하고 Streamlit 코드 작성해주세요....
