# 🌟 에테모 — 에겐/테토 페르소나 챗봇 구현 프로젝트

> **2025 XAI-ADV Session LLM 2 Team**
> *“How would Estrogen or Testosterone answer this question?”*

---

## 🎬 시연 PPT & 데모 영상

| 발표 PPT | 시연 영상 |
|:--:|:--:|
| [📄 PPT 보기](https://drive.google.com/file/d/1mW0Gg9x27R2f40_2LKrDH4HgadimJ7mm/view?usp=drive_link) | [▶️ 데모 영상 보기](https://youtu.be/...) |

---

## 🧠 프로젝트 개요

- **프로젝트명**: 에테모 (**E**strogen + **Te**stosterone + **Mo**del)
- **목표**: MZ세대가 SNS에서 사용하는 성격 유형, ‘에겐(Estrogen)’과 ‘테토(Testosterone)’를 LLM에 반영하여 각각의 페르소나를 일관되게 구현하는 챗봇 제작
- **핵심 과제**:
  - LLM에 페르소나를 내재화하는 파인튜닝 (SFT + DPO)
  - 서로 다른 성격의 답변을 안정적으로 생성하는 멀티페르소나 챗봇 구축

---

## 🧩 프로젝트 구조

```bash
📁 ethemo/
├── data/                  # 데이터 전처리 및 분할
│   ├── sft_data.csv
│   ├── dpo_data.json
│   └── eval_data.json
├── model/
│   ├── base/              # rtzr/ko-gemma-2-9b-it
│   ├── egen_model/
│   └── teto_model/
├── training/
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── config/
├── inference/
│   ├── demo.py
│   └── judge_eval.py
├── eval/
│   ├── human_eval_googleform/
│   └── llm_as_judge_result.csv
└── README.md
