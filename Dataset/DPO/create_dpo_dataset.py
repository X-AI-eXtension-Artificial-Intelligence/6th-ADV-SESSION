import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# --- Pandas 출력 설정 ---
pd.set_option('display.max_colwidth', None)

# --- OpenAI API Client ---
client = OpenAI(api_key="")

# 대화체에 대한 대답 구성을 위해 카카오 데이터셋에 적용
df = pd.read_csv('kakao_dataset.csv')

def merge_consecutive_utterances(df):
    merged_rows = []
    prev_speaker = None
    prev_txt = None
    buffer = ""

    for idx, row in df.iterrows():
        speaker = row["speaker"]
        utterance = row["utterance"]
        txt = row["txt"]

        # 이전과 speaker, txt 모두 같아야 병합
        if speaker == prev_speaker and txt == prev_txt:
            buffer += " " + utterance
        else:
            # 이전 발화 저장
            if prev_speaker is not None:
                merged_rows.append({
                    "speaker": prev_speaker,
                    "utterance": str(buffer).strip(),
                    "txt": prev_txt
                })
            # 새 발화 시작
            buffer = utterance
            prev_speaker = speaker
            prev_txt = txt

    # 마지막 행 처리
    if prev_speaker is not None:
        merged_rows.append({
            "speaker": prev_speaker,
            "utterance": str(buffer).strip(),
            "txt": prev_txt
        })

    return pd.DataFrame(merged_rows)

df_merged = merge_consecutive_utterances(df)

# 숫자 부분을 추출해 새로운 int 컬럼 추가
df_merged["txt_num"] = df_merged["txt"].str.extract(r"txt_(\d+)").astype(int)

# 범위 조건으로 필터링 (api 가격을 고려해 조금만 생성)
filtered_df = df_merged[(df_merged["txt_num"] >= 14000) & (df_merged["txt_num"] <= 14500)].copy()

# 시스템 프롬프트
ESTROGEN_STYLE_SYSTEM = """당신은 **다정**하고, **감성**적이며, 섬세한 **에스트로겐 스타일**의 사람입니다.
MBTI는 INFP입니다. 말투는 **부드럽고 공감하는 어조**입니다. 음식 취향은 건강식, 취미는 독서와 산책을 좋아합니다. 
감정 표현을 잘하며, 옷스타일은 따뜻한 색상의 내추럴 스타일입니다. 질문에 대해 친절하고 따뜻하게, **여성적으로 대답**하세요.
친한 친구와 대화하듯 반말로 편하게 얘기하고 간결하게 답하며 **이모티콘은 쓰지 말고** 느낌표나 물음표를 적극 활용하세요.
이전 발화 맥락을 고려하여 지금 발화에 대한 한국어 응답을 하세요."""

TESTOSTERONE_STYLE_SYSTEM = """당신은 **자신감**있고, **직설적**이며, **논리적**인 **테스토스테론 스타일**의 사람입니다.
MBTI는 ESTP입니다. 말투는 **간결하고 단호한 어조**입니다. 음식 취향은 고기류를 선호하고, 취미는 운동과 게임을 좋아합니다. 
감정보다 논리를 중시하며, 옷스타일은 깔끔하고 세련된 슈트 스타일입니다. 질문에 대해 논리적이고 직설적으로, **남성적으로 대답**하세요.
친한 친구와 대화하듯 반말로 편하게 얘기하고 간결하게 답하세요. 관심없는 얘기에는 꼭 다정하게 대답할 필요는 없습니다.
이전 발화 맥락을 고려하여 지금 발화에 대한 한국어 응답을 하세요."""

# 응답 생성 함수
def generate_response(system_prompt, full_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# DPO 데이터셋 생성
dpo_dataset = []

# 이전 질문을 얼마나 포함할지
max_history = 2  # 예: 최근 2개의 이전 질문까지 맥락 포함

# 전처리된 질문 리스트
question_list = filtered_df["utterance"].tolist()

for idx in tqdm(range(len(question_list)), desc="Generating DPO with context"):
    current_question = question_list[idx]
    
    # 이전 질문 추출
    history_questions = question_list[max(0, idx - max_history):idx]
    history_text = "\n".join([f"이전 발화: {q}" for q in history_questions])
    
    # 전체 프롬프트 구성
    full_prompt = (history_text + f"\n지금 발화: **{current_question}**").strip()

    try:
        estrogen_response = generate_response(ESTROGEN_STYLE_SYSTEM, full_prompt)
        time.sleep(1.5)
        testosterone_response = generate_response(TESTOSTERONE_STYLE_SYSTEM, full_prompt)
        time.sleep(1.5)

        dpo_dataset.append({
            "prompt": current_question,
            "context_prompt": full_prompt,
            "chosen": estrogen_response,
            "rejected": testosterone_response,
            "label": "에겐"
        })
        dpo_dataset.append({
            "prompt": current_question,
            "context_prompt": full_prompt,
            "chosen": testosterone_response,
            "rejected": estrogen_response,
            "label": "테토"
        })

    except Exception as e:
        print(f"[ERROR] at index {idx}: {current_question}\n{e}")
        continue

# 저장
with open("dpo_with_context.json", "w", encoding="utf-8") as f:
    json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)


df_all = pd.DataFrame(dpo_dataset)
df_all.to_csv('dpo_with_context.csv', index = False)