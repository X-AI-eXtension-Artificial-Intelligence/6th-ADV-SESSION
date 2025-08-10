import os
import json
import glob
import pandas as pd

# === 경로 설정 ===
IN_DIRS = ["PersonaData/Training", "PersonaData/Validation"]  # 입력 루트 폴더들
OUT_DIR = "Dataset"                                   # 출력 폴더
OUT_PATH = os.path.join(OUT_DIR, "utterance_persona.csv")

os.makedirs(OUT_DIR, exist_ok=True)                   # 출력 폴더 생성

# 빈값 제거 후 구분자로 연결
def _join(values, sep=" | "):
    return sep.join([str(v).strip() for v in values if v is not None and str(v).strip() != ""])

# info.personas → {persona_id: [persona entries...]} 매핑
def build_persona_map(info: dict) -> dict:
    persona_map = {}
    for p in info.get("personas", []) or []:
        pid = p.get("persona_id")
        persona_map[pid] = p.get("persona", []) or []
    return persona_map

# 단일 파일 처리 → row(dict) 리스트
def process_file(fp: str) -> list:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {}) or {}
    topic = info.get("topic")
    persona_map = build_persona_map(info)

    utts = data.get("utterances", []) or []
    n = len(utts)
    if n < 2:
        return rows  # 페어 생성 불가 시 스킵

    # i번째를 query, i+1번째를 response로 페어링
    for i in range(n - 1):
        q = utts[i]
        r = utts[i + 1]

        # 텍스트
        q_text = q.get("text")
        r_text = r.get("text")

        # response 화자의 persona 추출
        resp_pid = r.get("persona_id")
        persona_entries = persona_map.get(resp_pid, [])

        # persona 필드들을 각각 리스트로 수집
        profiles = [e.get("profile") for e in persona_entries]
        majors = [e.get("profile_major") for e in persona_entries]
        minors = [e.get("profile_minor") for e in persona_entries]

        # 레코드 구성 (CSV 친화적으로 문자열 연결)
        rows.append({
            "topic": topic,                                   # 대화 주제
            "query": q_text,                                  # 이전 발화
            "response": r_text,                               # 다음 발화
            "response_persona_profiles": _join(profiles),     # profile 전체
            "response_persona_profile_majors": _join(majors), # profile_major 전체
            "response_persona_profile_minors": _join(minors), # profile_minor 전체
        })

    return rows

# 메인 실행부
def main():
    all_rows = []
    # 두 입력 루트 폴더를 순회하며 모든 JSON 처리
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
