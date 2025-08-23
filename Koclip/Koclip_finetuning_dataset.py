# 라이브러리 정의
import sys
sys.path.append("/home/work/XAI_ADV")

import os
# 캐시 경로 설정 (XAI_ADV 폴더 내부로 통일)
os.environ["TRITON_CACHE_DIR"] = "/home/work/XAI_ADV/.triton"
os.environ["HF_HOME"] = "/home/work/XAI_ADV/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/work/XAI_ADV/.transformers_cache"

# 디렉토리 생성
for cache_dir in [os.environ["TRITON_CACHE_DIR"], os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"]]:
    os.makedirs(cache_dir, exist_ok=True)

from glob import glob
from PIL import Image
import torch
import pandas as pd
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import xml.etree.ElementTree as ET
from openai import OpenAI
from tqdm.auto import tqdm
import time
import json

# 1. 모델 및 프로세서 로드 (4bit 양자화)
MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,        # 4bit 양자화 (VRAM 절약)
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

# 2. OpenAI 클라이언트 초기화
client = OpenAI(api_key="")

# 3. Training 디렉토리 확인
training_dir = r"/home/work/XAI_ADV/Training"

# 4. [원천] 카테고리 탐색
source_folders = [f for f in os.listdir(training_dir) if f.startswith("[원천]")]
print(f"발견된 원천 폴더: {source_folders}")

results = []

# 전체 이미지 개수 미리 계산
total_images = 0
for src_cat in source_folders:
    src_cat_path = os.path.join(training_dir, src_cat)
    if os.path.exists(src_cat_path):
        for product_folder in os.listdir(src_cat_path):
            product_path = os.path.join(src_cat_path, product_folder)
            if os.path.isdir(product_path):
                jpg_count = sum(1 for f in os.listdir(product_path) if f.lower().endswith('.jpg'))
                total_images += jpg_count

print(f"총 처리할 이미지 개수: {total_images}")
overall_pbar = tqdm(total=total_images, desc="전체 진행률", unit="images")

# 5. 카테고리별 처리
for src_cat_idx, src_cat in enumerate(source_folders, 1):
    src_cat_path = os.path.join(training_dir, src_cat)
    label_cat = src_cat.replace("[원천]", "[라벨]")
    label_cat_path = os.path.join(training_dir, label_cat)

    if not os.path.exists(label_cat_path):
        print(f"[라벨] 카테고리 없음: {label_cat_path}")
        continue

    cat_images = 0
    for product_folder in os.listdir(src_cat_path):
        product_path = os.path.join(src_cat_path, product_folder)
        if os.path.isdir(product_path):
            jpg_count = sum(1 for f in os.listdir(product_path) if f.lower().endswith('.jpg'))
            cat_images += jpg_count

    print(f"\n{'='*60}")
    print(f"카테고리 [{src_cat_idx}/{len(source_folders)}]: {src_cat}")
    print(f"이 카테고리의 이미지 개수: {cat_images}")
    print(f"{'='*60}")

    cat_pbar = tqdm(total=cat_images, desc=f"카테고리 {src_cat_idx}",
                    leave=False, unit="images", position=1)

    for product_folder in os.listdir(src_cat_path):
        product_path = os.path.join(src_cat_path, product_folder)
        if not os.path.isdir(product_path):
            continue

        label_product_path = os.path.join(label_cat_path, product_folder)
        if not os.path.exists(label_product_path):
            tqdm.write(f"[라벨] 상품폴더 없음: {label_product_path}")
            continue

        image_files = [os.path.join(product_path, f)
                       for f in os.listdir(product_path) if f.lower().endswith('.jpg')]

        for image_path in image_files:
            start_time = time.time()
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            xml_path = os.path.join(label_product_path, f"{base_name}_meta.xml")

            cat_pbar.set_description(f"처리 중: {base_name[:20]}...")
            overall_pbar.set_description(f"전체: {base_name[:15]}...")

            try:
                # 이미지 캡션 생성
                batch = [{
                    "image": [Image.open(image_path).convert("RGB")],
                    "conv": [
                        {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                        {"role": "user", "content": "<image>"},
                        {"role": "user", "content": "이 사진에 대해 전체적으로 설명해줘."},
                    ]
                }]

                inputs = processor.batch_encode_collate(
                    batch, padding_side="left", add_generation_prompt=True, max_length=2048
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                gen_kwargs = {
                    "max_new_tokens": 256,
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "num_beams": 1,
                    "do_sample": False,
                }

                gens = model.generate(**inputs, **gen_kwargs)
                image_caption = processor.tokenizer.batch_decode(gens, skip_special_tokens=True)[0]

                # XML 메타데이터 읽기
                metadata = {}
                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        div_cd = root.find('div_cd')
                        if div_cd is not None:
                            for tag in ["item_cd", "item_no", "div_l", "div_m", "div_s", "div_n", "comp_nm", "img_prod_nm", "volume"]:
                                el = div_cd.find(tag)
                                if el is not None:
                                    metadata[tag] = el.text
                    except Exception as e:
                        tqdm.write(f"XML 파싱 오류 ({base_name}): {e}")

                # GPT 요약
                prompt = f"""
                다음은 상품 메타데이터와 이미지 캡션입니다.
                이미지 캡션에 나오는 색상, 모양, 크기 등 시각적 정보까지 포함해,
                상품의 주요 특징과 브랜드, 용량, 용도, 디자인 등을 모두 반영한 자연스럽고 구체적인 한 문장으로 요약해주세요.

                [메타데이터]
                {metadata}

                [이미지 캡션]
                {image_caption}
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )

                label_sentence = response.choices[0].message.content

                results.append({
                    "image_path": image_path,
                    "caption": label_sentence,
                    "image_caption": image_caption,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "category": src_cat,
                    "product_folder": product_folder
                })

                elapsed_time = time.time() - start_time
                if len(results) % 10 == 0:
                    tqdm.write(f"\n처리 완료: {base_name}")
                    tqdm.write(f"이미지 캡션: {image_caption[:100]}...")
                    tqdm.write(f"최종 라벨: {label_sentence}")
                    tqdm.write(f"처리 시간: {elapsed_time:.2f}초")

            except Exception as e:
                tqdm.write(f"오류 발생 ({base_name}): {e}")
                continue

            cat_pbar.update(1)
            overall_pbar.update(1)

    cat_pbar.close()
    tqdm.write(f"카테고리 '{src_cat}' 처리 완료")

overall_pbar.close()

# 6. CSV 저장
output_csv_path = os.path.join(training_dir, "koclip_finetuning_dataset.csv")
df = pd.DataFrame(results)
df = df[["image_path", "caption", "image_caption", "metadata", "category", "product_folder"]]
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")