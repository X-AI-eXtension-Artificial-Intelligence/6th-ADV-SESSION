from huggingface_hub import snapshot_download
from pathlib import Path
import os

repo_id   = "rtzr/ko-gemma-2-9b-it"
base_dir  = "ko-gemma-2-9b-it"  # 저장 폴더
cache_dir = base_dir                                      # 캐시를 저장 폴더와 동일하게

Path(base_dir).mkdir(parents=True, exist_ok=True)

# 캐시 환경변수도 동일 폴더로 고정(다른 위치에 쌓이지 않도록)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"]  = cache_dir
# 심볼릭 링크 경고 숨김(선택)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,             # ← 캐시를 저장폴더로
    local_dir=base_dir,              # ← 출력도 같은 폴더
    local_dir_use_symlinks=True,     # ← 복사 X (공간 추가 소모 최소화)
    resume_download=True,            # 중단 이어받기
    force_download=False             # 불필요한 강제 재다운로드 X
)
