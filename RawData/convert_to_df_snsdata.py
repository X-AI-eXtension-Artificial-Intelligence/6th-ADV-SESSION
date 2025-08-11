
import os
import json
import glob
import pandas as pd

# === 설정 ===
ROOT_DIRS = {
    "training": "RawData/Training",      # 학습 데이터 폴더
    "validation": "RawData/Validation"   # 검증 데이터 폴더
}
OUT_DIR = "Dataset/SFT"                      # 결과 저장 폴더
OUT_PATH = os.path.join(OUT_DIR, "utterance_sns.csv")

os.makedirs(OUT_DIR, exist_ok=True)      # 결과 폴더 생성

# 화자 정보 매핑 생성 함수
def build_speaker_meta(info_speaker: dict) -> dict:
    # speakerA/B/C 각각의 id, 성별, 나이 정보를 매핑
    meta = {}
    for code in ("A", "B", "C"):
        meta[f"speaker{code}"] = {
            "sex": info_speaker.get(f"speaker{code}Sex"),
            "age": info_speaker.get(f"speaker{code}Age"),
        }
    return meta

# 개별 JSON 파일 처리 함수
def process_file(fp: str, dataset_type: str) -> list:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    topic = info.get("topic")
    keyword = info.get("keyword")

    speaker_info = build_speaker_meta(info.get("speaker", {}))
    utts = data.get("utterances", []) or []
    n = len(utts)

    if n < 2:
        return rows  # 페어 생성 불가 시 스킵

    # i번째 발화를 query, i+1번째 발화를 response로 매핑
    for i in range(n - 1):
        q = utts[i]
        r = utts[i + 1]

        resp_spk_code = r.get("speaker")              # 응답 화자 코드
        resp_meta = speaker_info.get(resp_spk_code, {})

        row = {
            "topic": topic,                           # 대화 주제
            "keyword": keyword,                       # 키워드
            "query": q.get("text"),                   # 이전 발화
            "response": r.get("text"),                # 다음 발화
            "response_speaker": resp_spk_code,        # 응답 화자 코드
            "response_sex": resp_meta.get("sex"),     # 응답 화자 성별
            "response_age": resp_meta.get("age"),     # 응답 화자 나이
        }
        rows.append(row)

    return rows

# 메인 실행부
def main():
    all_rows = []

    # training / validation 각각 처리
    for dataset_type, root_dir in ROOT_DIRS.items():
        files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
        for fp in files:
            try:
                all_rows.extend(process_file(fp, dataset_type))
            except Exception as e:
                print(f"[WARN] Failed to process {fp}: {e}")

    if not all_rows:
        print("No pairs were found. Check your directories or file formats.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")  # 전체 CSV 저장
    print(f"Saved: {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()