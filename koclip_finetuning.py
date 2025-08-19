# 라이브러리 정의
import os, math, torch, random
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image

from datasets import Dataset
from transformers import (
    VisionTextDualEncoderModel,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 설정 변수 정의
BASE_DIR = "/home/work/XAI_ADV"
ORIG_CSV = os.path.join(BASE_DIR, "Training", "koclip_finetuning_dataset.csv")

# 학습에 사용될 1800개와, 나머지 벡터 DB 저장용 216개 분리
TRAIN_CSV = os.path.join(BASE_DIR, "Training", "train.csv")
INFER_CSV = os.path.join(BASE_DIR, "Training", "inference.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "koclip_lora_adapter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); random.seed(SEED)

# CSV 분리 (처음 1800개 → train.csv, 나머지 216개 → inference.csv)
df = pd.read_csv(ORIG_CSV)

train_df = df.iloc[:1800].copy()
infer_df = df.iloc[1800:].copy()

train_df.to_csv(TRAIN_CSV, index=False)
infer_df.to_csv(INFER_CSV, index=False)

print(f"Train CSV saved: {TRAIN_CSV}, shape={train_df.shape}")
print(f"Inference CSV saved: {INFER_CSV}, shape={infer_df.shape}")

# 모델/프로세서 로드
REPO = "koclip/koclip-base-pt"  # pytorch 버전 KoCLIP
processor = AutoProcessor.from_pretrained(REPO)
model = VisionTextDualEncoderModel.from_pretrained(REPO)

# projection & logit_scale만 업데이트 허용
for p in model.parameters():
    p.requires_grad_(False)

model.text_projection.requires_grad_(True)
model.visual_projection.requires_grad_(True)
model.logit_scale.requires_grad_(True)

# 텍스트 타워에만 LoRA 주입
text_lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

model.text_model = get_peft_model(model.text_model, text_lora_cfg)

model.to(DEVICE)
model.train()

# Dataset 준비 
df = pd.read_csv(TRAIN_CSV)

df = df[["image_path", "caption"]].copy()
df["text"] = df["caption"].astype(str)

# HuggingFace Dataset으로 변환
dataset = Dataset.from_pandas(df)

# train/val split (90:10)
split = dataset.train_test_split(test_size=0.1, seed=SEED)
train_ds, val_ds = split["train"], split["test"]

# collate_fn 정의
@dataclass
class CLIPBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor

def collate_fn(examples):
    images = [Image.open(p).convert("RGB") for p in [ex["image_path"] for ex in examples]]
    texts = [ex["text"] for ex in examples]
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch["pixel_values"],
    }

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

# 학습 루프
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

global_step = 0
model.zero_grad(set_to_none=True)

for epoch in range(EPOCHS):
    running = 0.0
    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            out = model(**batch, return_loss=True)
            loss = out.loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        running += out.loss.item()
        if step % (GRAD_ACCUM * 10) == 0:
            print(f"[epoch {epoch+1}] step {step:5d} | loss {running/(GRAD_ACCUM*10):.4f}")
            running = 0.0

# 모델 저장
FULL_DIR = os.path.join(OUTPUT_DIR, "koclip_finetuning_model")
os.makedirs(FULL_DIR, exist_ok=True)

model.save_pretrained(FULL_DIR)
processor.save_pretrained(FULL_DIR)

print("Saved FULL model to:", FULL_DIR)
