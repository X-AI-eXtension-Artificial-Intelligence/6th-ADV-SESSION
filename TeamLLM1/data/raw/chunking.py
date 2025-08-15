# -*- coding: utf-8 -*-
"""
chunking.py
- 입력 CSV(법령명, 조문 번호, 조문 내용) → 조 단위 청킹 후 [법령_조문, 법령_텍스트] 저장
- main()은 .env만 읽고 chunk_law_dataframe(df) 호출

.env 예시:
CHUNKING_INPUT_CSV=raw_law.csv
CHUNKING_OUTPUT_CSV=law_db.csv
"""

import re
import os
import sys
import pandas as pd
import unicodedata

try:
    from dotenv import load_dotenv  # 선택
    load_dotenv()
except Exception:
    pass


# --------- 유틸 함수들 ---------
def normalize(s):
    return unicodedata.normalize("NFKC", str(s)).strip()

_CIRCLED_MAP = dict(zip("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", range(1, 21)))
def decircle(s):
    s = str(s)
    for k, v in _CIRCLED_MAP.items():
        s = s.replace(k, str(v))
    return s

_JO_PAT   = re.compile(r'(\d+)\s*조')
_HANG_PAT = re.compile(r'(\d+)\s*항')
_HO_PAT   = re.compile(r'(\d+)\s*호')

def parse_numbers(s):
    s = str(s)
    jo_m   = _JO_PAT.search(s)
    hang_m = _HANG_PAT.search(s)
    ho_m   = _HO_PAT.search(s)
    jo   = int(jo_m.group(1)) if jo_m else None
    hang = int(hang_m.group(1)) if hang_m else 0
    ho   = int(ho_m.group(1)) if ho_m else 0
    return jo, hang, ho

def strip_header(text):
    return re.sub(r'^제\s*\d+\s*조.*?\n', '', str(text), flags=re.S)

def dedup_sentences(s):
    sents = re.split(r'\s*(?<=[\.!?])\s+', str(s).strip())
    out, seen = [], set()
    for t in sents:
        t = t.strip()
        if t and t not in seen:
            out.append(t); seen.add(t)
    return " ".join(out)

def extract_header(s):
    s = str(s)
    m = re.search(r'(제\s*\d+\s*조)\s*\((.*?)\)', s)
    if m:
        return f"{m.group(1).replace(' ', '')} · {m.group(2)}"
    m2 = re.search(r'(제\s*\d+\s*조)', s)
    return m2.group(1).replace(' ', '') if m2 else None

# --------- 핵심 청킹 함수 ---------
def chunk_law_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 df는 반드시 [법령명, 조문 번호, 조문 내용] 컬럼 포함.
    반환 df는 [법령_조문, 법령_텍스트].
    """
    for col in ['법령명', '조문 번호', '조문 내용']:
        if col not in df.columns:
            raise ValueError(f"입력 DataFrame에 필요한 컬럼이 없습니다: {col}")

    law = df[['법령명', '조문 번호', '조문 내용']].copy()

    # 0) 정규화
    law["법령명"] = law["법령명"].map(normalize)
    law["조문 번호"] = law["조문 번호"].map(normalize)
    law["조문 내용"] = law["조문 내용"].map(normalize)

    # 1) ①②… → 1,2,…
    law["조문 번호_n"] = law["조문 번호"].map(decircle)

    # 2) 조/항/호 파싱
    law["조"], law["항"], law["호"] = zip(*law["조문 번호_n"].map(parse_numbers))

    # 3) 헤더 제거
    law["조문 본문"] = law["조문 내용"].map(strip_header)

    # 4) 정렬 및 본문 합치기(중복문장 제거)
    law = law.dropna(subset=["법령명", "조"])
    law_sorted = law.sort_values(["법령명", "조", "항", "호"])

    agg = (
        law_sorted.groupby(["법령명", "조"], as_index=False)
        .agg({
            "조문 내용": "first",
            "조문 본문": lambda x: dedup_sentences(" ".join(map(str, x)))
        })
    )

    # 5) 헤더 추출
    agg["조문 헤더"] = agg["조문 내용"].map(extract_header)

    # 6) 최종 텍스트
    agg["법령_조문"] = agg.apply(lambda r: f"{r["법령명"]} {int(r['조'])}조", axis=1)
    agg["법령_텍스트"] = agg.apply(
        lambda r: f"[법령] {r["법령명"]} [조문] {r['조문 헤더'] or str(int(r['조']))+'조'}\n{r['조문 본문']}".strip(),
        axis=1
    )

    return agg[["법령_조문", "법령_텍스트"]].copy()

# --------- main: .env만 사용 ---------
def main():
    in_path  = os.getenv("CHUNKING_INPUT_CSV", None)
    out_path = os.getenv("CHUNKING_OUTPUT_CSV", "law_db.csv")

    if not in_path or not os.path.exists(in_path):
        print(f"[오류] 입력 CSV가 존재하지 않습니다: {in_path}")
        sys.exit(1)

    if os.path.abspath(in_path) == os.path.abspath(out_path):
        print("[오류] 입력과 출력 경로가 같습니다. CHUNKING_OUTPUT_CSV를 다르게 지정하세요.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(df.head(5))
    try:
        law_db = chunk_law_dataframe(df)
    except Exception as e:
        print(f"[오류] 청킹 중 오류: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    law_db.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[완료] law_db CSV 저장: {out_path} (행수: {len(law_db)})")

if __name__ == "__main__":
    main()
