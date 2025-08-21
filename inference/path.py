import os, sys

paths = {
    "PIP_CACHE_DIR": "/cache/pip",
    "PYTHONUSERBASE": "/.local_lib",
    "HF_HOME": "/hf",
    "TRANSFORMERS_CACHE": "/cache/hf_transformers",
    "HF_DATASETS_CACHE": "/cache/hf_datasets",
    "TORCH_HOME": "/cache/torch",
    "XDG_CACHE_HOME": "/cache",
    "CUDA_CACHE_PATH": "/cache/ccache",
    "TMPDIR": "/tmp",
}

# 환경변수 적용
for k, v in paths.items():
    os.environ[k] = v
    # 폴더가 없을 때만 생성
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)

# PYTHONPATH 확장
sys.path.insert(
    0,
    "/.local_lib/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
