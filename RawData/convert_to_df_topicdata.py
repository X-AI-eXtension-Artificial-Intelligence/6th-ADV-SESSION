import os
import json
import glob
import pandas as pd

# === 경로 설정 ===
IN_DIRS = ["TopicData/Training", "TopicData/Validation"]  # 입력 루트 폴더들
OUT_DIR = "Dataset/SFT"                                   # 출력 폴더
OUT_PATH = os.path.join(OUT_DIR, "utterance_topic.csv")

os.makedirs(OUT_DIR, exist_ok=True)                   # 출력 폴더 생성

def process_file(fp: str) -> list:
    """단일 JSON 파일 처리 → row(dict) 리스트 반환"""
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # info는 리스트이므로 항목별로 순회
    infos = data.get("info", []) or []
    for item in infos:
        # 메타 필드 추출
        mediatype = item.get("mediatype")
        medianame = item.get("medianame")
        annotations = item.get("annotations", {}) or {}
        subject = annotations.get("subject")

        # 대화 라인
        lines = (annotations.get("lines") or [])
        n = len(lines)
        if n < 2:
            continue  # 페어 불가 시 스킵

        # i번째를 query, i+1번째를 response로 페어링
        for i in range(n - 1):
            q = lines[i]
            r = lines[i + 1]

            # 텍스트
            q_text = q.get("norm_text") or q.get("text")
            r_text = r.get("norm_text") or r.get("text")

            # response 화자 성별/나이 (최대 3명 화자 고려)
            r_speaker = r.get("speaker", {}) or {}
            r_sex = r_speaker.get("sex")
            r_age = r_speaker.get("age")

            # 레코드 구성
            rows.append({
                "query": q_text,
                "response": r_text,
                "response_sex": r_sex,
                "response_age": r_age,
                "mediatype": mediatype,
                "medianame": medianame,
                "subject": subject,
            })

    return rows

def main():
    all_rows = []

    # 입력 루트 폴더 각각 처리
    for root in IN_DIRS:
        files = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
        for fp in files:
            try:
                all_rows.extend(process_file(fp))
            except Exception as e:
                # 손상 파일 등은 경고만 출력하고 계속 진행
                print(f"[WARN] Failed to process {fp}: {e}")

    if not all_rows:
        print("No pairs found. Check input directories or JSON format.")
        return

    # DataFrame 생성 및 저장
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()