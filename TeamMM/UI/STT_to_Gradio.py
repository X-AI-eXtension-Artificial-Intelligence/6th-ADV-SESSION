# 코드 블록 6개로 구성 
# #설정에서 자신의 환경에 맞게 설정 

# 코드 블록 1: 기본 설정 및 Vision 모델
import os
import ast
import json
import time
import xml.etree.ElementTree as ET
from glob import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gradio as gr
import gc

# 설정
OPENAI_API_KEY = ""  # 실제 API 키로 교체
DATA_FOLDER = ""
PRODUCT_CSV_PATH = "/product_labels_final.csv"
WHISPER_MODEL_NAME = "SungBeom/whisper-small-ko"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
VISION_MODEL_NAME = "kakaocorp/kanana-1.5-v-3b-instruct"
GPT_MODEL_NAME = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.25

# 메모리 관리
def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_vision_model():
    """Vision 모델 로드"""
    print("Vision 모델 로드 중...")
    cleanup_memory()
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            VISION_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True)
        print("✅ Vision 모델 로드 완료!")
        return model, processor
    except Exception as e:
        print(f"❌ Vision 모델 로드 실패: {e}")
        return None, None

def generate_image_caption(image_path, model, processor):
    """이미지 캡션 생성"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        batch = [{
            "image": [image],
            "conv": [
                {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": "이 사진에 대해 전체적으로 설명해줘."},
            ]
        }]
        
        inputs = processor.batch_encode_collate(
            batch,
            padding_side="left",
            add_generation_prompt=True,
            max_length=8192
        )
        
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(device=model_device, dtype=model_dtype)
                else:
                    processed_inputs[key] = value.to(device=model_device)
            else:
                processed_inputs[key] = value
        
        gen_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        }
        
        with torch.no_grad():
            gens = model.generate(**processed_inputs, **gen_kwargs)
        
        caption = processor.tokenizer.batch_decode(gens, skip_special_tokens=True)[0]
        cleanup_memory()
        return caption
        
    except Exception as e:
        print(f"캡션 생성 오류: {e}")
        cleanup_memory()
        return f"이미지 처리 실패: {os.path.basename(image_path)}"

def parse_xml_metadata(xml_path):
    """XML 메타데이터 파싱"""
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
            print(f"XML 파싱 오류: {e}")
    return metadata

print("코드 블록 1 완료!")


# 코드 블록 2: OpenAI 관련 함수

def init_openai_client():
    """OpenAI 클라이언트 초기화"""
    return OpenAI(api_key=OPENAI_API_KEY)

def generate_product_label_with_gpt(image_caption, metadata, client):
    """GPT를 사용한 상품 라벨 생성"""
    prompt = f"""
다음은 상품 메타데이터와 이미지 캡션입니다.
이미지 캡션에 나오는 색상, 모양, 크기 등 시각적 정보까지 포함해,
상품의 주요 특징과 브랜드, 용량, 용도, 디자인 등을 모두 반영한 자연스럽고 구체적인 한 문장으로 요약해주세요.

[메타데이터]
{metadata}

[이미지 캡션]
{image_caption}
"""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT 라벨 생성 오류: {e}")
        return f"라벨 생성 실패: {metadata.get('img_prod_nm', 'Unknown')}"

def extract_entities_with_features(text, client):
    """음성에서 상품 엔티티 추출"""
    instruction = f'''
다음은 사용자가 말한 문장입니다.
이 문장에서 장바구니에 담을 상품명과 주요 특징(용량, 수량 등)을 함께 포함해서,
각 항목을 '상품명(특징)' 형태로 뽑아주세요. 용량이 없으면 생략, 수량이 없으면 1개로 표시해주세요.
여러 개는 쉼표로 구분된 리스트 형태로 간결하게 뽑아주세요.

문장: "{text}"

예시)
입력: 오뚜기 카레 분말 하나랑, 햇반 작은 거 두 개 넣어줄래?
출력: 오뚜기 카레 분말 (1개), 햇반 (작은 거, 2개)
'''
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "user", "content": instruction}],
            temperature=0,
            max_tokens=200
        )
        result_text = response.choices[0].message.content.strip()
        entities = [e.strip() for e in result_text.split(",") if e.strip()]
        return entities
    except Exception as e:
        print(f"엔티티 추출 오류: {e}")
        return [text]

print("코드 블록 2 완료!")



# 코드 블록 3: 제품 라벨 생성

def create_product_labels_batch(image_files, batch_size=3):
    """배치로 상품 라벨 데이터 생성"""
    vision_model, vision_processor = load_vision_model()
    openai_client = init_openai_client()
    
    if vision_model is None:
        return []
    
    product_data = []
    print(f"총 {len(image_files)}개 이미지 처리 시작...")
    
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        print(f"\n=== 배치 {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ===")
        
        for idx, image_path in enumerate(batch_files):
            base_name = os.path.basename(image_path).replace(".jpg", "")
            xml_path = os.path.join(DATA_FOLDER, f"{base_name}_meta.xml")
            
            print(f"[{batch_start + idx + 1}/{len(image_files)}] 처리 중: {base_name}")
            
            try:
                image_caption = generate_image_caption(image_path, vision_model, vision_processor)
                metadata = parse_xml_metadata(xml_path)
                label_sentence = generate_product_label_with_gpt(image_caption, metadata, openai_client)
                
                product_data.append({
                    'base_name': base_name,
                    'image_file': image_path,
                    'image_caption': image_caption,
                    'metadata': json.dumps(metadata, ensure_ascii=False),
                    'label_sentence': label_sentence
                })
                
                print(f"    완료: {label_sentence[:50]}...")
                
            except Exception as e:
                print(f"    오류 발생 ({base_name}): {e}")
                continue
        
        cleanup_memory()
    
    return product_data

# 테스트 실행 (처음 5개 이미지만)
if os.path.exists(DATA_FOLDER) and OPENAI_API_KEY != "your-openai-api-key-here":
    image_files = glob(os.path.join(DATA_FOLDER, "*.jpg"))[:5]  # 처음 5개만 테스트
    
    if image_files:
        print("상품 라벨 생성 테스트 시작...")
        product_data = create_product_labels_batch(image_files, batch_size=2)
        
        # 결과 확인
        print(f"\n생성된 라벨 데이터: {len(product_data)}개")
        for data in product_data:
            print(f"- {data['base_name']}: {data['label_sentence']}")
        
        # CSV로 저장 (테스트)
        if product_data:
            df = pd.DataFrame(product_data)
            test_csv_path = "/content/product_labels_test.csv"
            df.to_csv(test_csv_path, index=False, encoding='utf-8')
            print(f"\n테스트 CSV 저장 완료: {test_csv_path}")
    else:
        print("테스트할 이미지가 없습니다.")
else:
    print("데이터 폴더가 없거나 OpenAI API 키가 설정되지 않았습니다.")

print("코드 블록 3 완료!")



# 코드 블록 4: 음성 인식 모델

def load_whisper_model():
    """Whisper 모델 로드"""
    print("Whisper 모델 로드 중...")
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Whisper 모델 로드 완료!")
    return whisper_model

def process_voice_input(audio_file, whisper_model, openai_client):
    """음성 파일 처리"""
    if audio_file is None:
        return "음성을 입력해주세요.", []
    
    try:
        # STT
        result = whisper_model(audio_file)
        recognized_text = result["text"]
        
        # 엔티티 추출
        entities = extract_entities_with_features(recognized_text, openai_client)
        
        return recognized_text, entities
    except Exception as e:
        return f"음성 처리 오류: {str(e)}", []

# 테스트용 텍스트 엔티티 추출
if OPENAI_API_KEY != "your-openai-api-key-here":
    print("엔티티 추출 테스트...")
    openai_client = init_openai_client()
    
    test_sentences = [
        "오뚜기 카레 분말 하나랑 햇반 작은 거 두 개 넣어줄래?",
        "파란색 포카칩이랑 보라색깔 포키 하나 담아줘",
        "바나나 우유 하나랑 초코우유 200ml 두 개 담아줘"
    ]
    
    for sentence in test_sentences:
        entities = extract_entities_with_features(sentence, openai_client)
        print(f"입력: {sentence}")
        print(f"추출: {entities}")
        print()

print("코드 블록 4 완료!")



# 코드 블록 5: 유사도 검색 모듈

def load_embedding_model():
    """임베딩 모델 로드"""
    print("임베딩 모델 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ 임베딩 모델 로드 완료!")
    return embedding_model

def load_product_data():
    """상품 데이터 로드"""
    test_csv_path = "/content/product_labels_test.csv"
    
    if os.path.exists(PRODUCT_CSV_PATH):
        print("기존 상품 데이터 로드 중...")
        product_data = pd.read_csv(PRODUCT_CSV_PATH)
    elif os.path.exists(test_csv_path):
        print("테스트 데이터 사용...")
        product_data = pd.read_csv(test_csv_path)
    else:
        print("상품 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 메타데이터 파싱 및 필드 추출
    def parse_metadata(metadata_str):
        try:
            if pd.isna(metadata_str) or metadata_str == '{}':
                return {}
            return ast.literal_eval(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except:
            return {}
    
    product_data['parsed_metadata'] = product_data['metadata'].apply(parse_metadata)
    
    def extract_field(row, field_name):
        try:
            return row['parsed_metadata'].get(field_name, '')
        except:
            return ''
    
    product_data['brand'] = product_data.apply(lambda x: extract_field(x, 'comp_nm'), axis=1)
    product_data['product_name'] = product_data.apply(lambda x: extract_field(x, 'img_prod_nm'), axis=1)
    product_data['category_medium'] = product_data.apply(lambda x: extract_field(x, 'div_m'), axis=1)
    product_data['category_small'] = product_data.apply(lambda x: extract_field(x, 'div_s'), axis=1)
    product_data['volume'] = product_data.apply(lambda x: extract_field(x, 'volume'), axis=1)
    
    print(f"상품 데이터 로드 완료: {len(product_data)}개")
    return product_data

def find_similar_products(entities, product_data, embedding_model):
    """엔티티와 유사한 상품 검색 - 엔티티당 최고 유사도 1개씩만"""
    if not entities or product_data.empty:
        return "검색할 데이터가 없습니다.", []
    
    try:
        all_results = []
        
        # 상품 라벨들 미리 임베딩 (효율성)
        product_labels = product_data['label_sentence'].tolist()
        product_embeddings = embedding_model.encode(product_labels)
        
        for entity in entities:
            print(f"검색 중: {entity}")
            
            # 엔티티 임베딩
            entity_embedding = embedding_model.encode([entity])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(entity_embedding, product_embeddings)[0]
            
            # 가장 높은 유사도 상품 1개만 선택
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity > SIMILARITY_THRESHOLD:
                row = product_data.iloc[best_idx]
                all_results.append({
                    'query_entity': entity,
                    'matched_product': row['label_sentence'],
                    'brand': row.get('brand', 'N/A'),
                    'category': f"{row.get('category_medium', '')} > {row.get('category_small', '')}".strip(' > '),
                    'volume': row.get('volume', 'N/A'),
                    'similarity_score': float(best_similarity),
                    'image_path': row['image_file'],
                    'base_name': row['base_name']
                })
        
        if not all_results:
            return "유사한 상품을 찾을 수 없습니다.", []
        
        # 결과 포맷팅
        result_text = f"찾은 상품 {len(all_results)}개:\n\n"
        images_to_show = []
        
        for i, result in enumerate(all_results):
            result_text += f"{i+1}. {result['matched_product']}\n"
            result_text += f"   매칭: [{result['query_entity']}]\n"
            result_text += f"   브랜드: {result['brand']}\n"
            result_text += f"   카테고리: {result['category']}\n"
            result_text += f"   용량: {result['volume']}\n"
            result_text += f"   유사도: {result['similarity_score']:.3f}\n\n"
            
            # 이미지 로드
            try:
                if os.path.exists(result['image_path']):
                    img = Image.open(result['image_path']).convert('RGB')
                    images_to_show.append(img)
            except Exception as e:
                print(f"이미지 로드 오류: {e}")
        
        return result_text, images_to_show
    
    except Exception as e:
        return f"상품 검색 오류: {str(e)}", []

# 유사도 검색 테스트
if os.path.exists(PRODUCT_CSV_PATH) or os.path.exists("/content/product_labels_test.csv"):
    print("유사도 검색 테스트...")
    
    # 모델 로드
    embedding_model = load_embedding_model()
    product_data = load_product_data()
    
    if not product_data.empty:
        # 테스트 엔티티들
        test_entities = [
            "오뚜기 카레 분말 (1개)",
            "파란색 포카칩 (1개)",
            "바나나 우유 (1개)"
        ]
        
        print(f"상품 데이터: {len(product_data)}개")
        print(f"테스트 엔티티: {test_entities}")
        
        # 유사도 검색 테스트
        result_text, result_images = find_similar_products(test_entities, product_data, embedding_model)
        print("\n=== 검색 결과 ===")
        print(result_text)
        print(f"결과 이미지 수: {len(result_images)}")
    else:
        print("상품 데이터가 비어있습니다.")
else:
    print("상품 데이터가 없습니다. 먼저 3번 셀을 실행하여 상품 라벨을 생성하세요.")

print("코드 블록 5 완료!")




# 코드 블록 6: Gradio 인터페이스

def create_voice_shopping_app():
    """통합 음성 쇼핑 Gradio 앱"""
    global whisper_model, embedding_model, product_data, openai_client
    
    print("모든 모델 로딩 중...")
    whisper_model = load_whisper_model()
    embedding_model = load_embedding_model()
    product_data = load_product_data()
    openai_client = init_openai_client()
    print("✅ 모든 모델 로딩 완료!")
    
    def process_voice_shopping(audio_file=None, microphone_audio=None):
        """음성 쇼핑 통합 처리"""
        start_time = time.time()
        
        # 음성 처리
        if audio_file is not None:
            recognized_text, entities = process_voice_input(audio_file, whisper_model, openai_client)
        elif microphone_audio is not None:
            recognized_text, entities = process_voice_input(microphone_audio, whisper_model, openai_client)
        else:
            return "음성을 입력해주세요.", "엔티티가 추출되지 않았습니다.", []
        
        if isinstance(recognized_text, str) and "오류" in recognized_text:
            return recognized_text, "엔티티 추출 실패", []
        
        # 엔티티 정보 표시
        entities_text = "추출된 상품 정보:\n"
        for i, entity in enumerate(entities):
            entities_text += f"{i+1}. {entity}\n"
        
        # 상품 검색
        product_results, product_images = find_similar_products(entities, product_data, embedding_model)
        
        processing_time = time.time() - start_time
        
        # 최종 결과 조합
        final_text = f"인식된 음성: {recognized_text}\n\n"
        final_text += f"처리 시간: {processing_time:.2f}초\n\n"
        final_text += product_results
        
        return final_text, entities_text, product_images
    
    # Gradio 인터페이스 구성
    with gr.Blocks(title="음성 쇼핑 시스템", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🛒 AI 음성 쇼핑 시스템
        
        **음성으로 상품을 주문하고 AI가 관련 상품을 찾아드립니다!**
        
        ### 사용법
        1. 마이크로 음성을 녹음하거나 음성 파일을 업로드하세요
        2. "검색하기" 버튼을 클릭하세요
        3. AI가 음성을 인식하고 관련 상품을 찾아서 보여줍니다
        
        ### 말하기 예시
        - "오뚜기 카레 분말 하나랑 농심 신라면 두 개"
        - "코카콜라 500ml 하나랑 새우깡 담아줘"
        - "바나나 우유 작은 걸로 하나, 초코파이 큰 거 두 개"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 음성 입력")
                
                microphone_input = gr.Audio(
                    label="마이크로 녹음하기",
                    type="filepath"
                )
                
                audio_file_input = gr.Audio(
                    label="음성 파일 업로드",
                    type="filepath"
                )
                
                with gr.Row():
                    search_btn = gr.Button("🔍 검색하기", variant="primary", size="lg")
                    clear_btn = gr.Button("🧹 초기화", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 처리 결과")
                
                result_text = gr.Textbox(
                    label="음성 인식 및 상품 검색 결과",
                    lines=12,
                    max_lines=20
                )
                
                entities_text = gr.Textbox(
                    label="추출된 상품 정보",
                    lines=6,
                    max_lines=10
                )
        
        gr.Markdown("### 🛍️ 찾은 상품들")
        product_gallery = gr.Gallery(
            label="관련 상품 이미지",
            show_label=True,
            columns=3,
            rows=2,
            height=400,
            allow_preview=True
        )
        
        # 이벤트 바인딩
        search_btn.click(
            fn=process_voice_shopping,
            inputs=[audio_file_input, microphone_input],
            outputs=[result_text, entities_text, product_gallery]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, "", "", []),
            outputs=[audio_file_input, microphone_input, result_text, entities_text, product_gallery]
        )
    
    return app

# 앱 실행
if __name__ == "__main__":
    # 필수 설정 검증
    if OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ OpenAI API 키를 설정해주세요!")
    elif not os.path.exists(DATA_FOLDER):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_FOLDER}")
    elif not (os.path.exists(PRODUCT_CSV_PATH) or os.path.exists("/content/product_labels_test.csv")):
        print("❌ 상품 데이터가 없습니다. 먼저 3번 셀을 실행하여 라벨 데이터를 생성하세요.")
    else:
        print("✅ 음성 쇼핑 시스템 시작!")
        app = create_voice_shopping_app()
        app.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

# 간단 테스트용 함수
def quick_test():
    """빠른 기능 테스트"""
    print("=== 빠른 기능 테스트 ===")
    
    # 텍스트 기반 엔티티 추출 테스트
    test_text = "바나나 우유 하나랑 초코파이 두 개 담아줘"
    print(f"테스트 텍스트: {test_text}")
    
    if OPENAI_API_KEY != "your-openai-api-key-here":
        client = init_openai_client()
        entities = extract_entities_with_features(test_text, client)
        print(f"추출된 엔티티: {entities}")
        
        # 상품 검색 테스트
        if os.path.exists(PRODUCT_CSV_PATH) or os.path.exists("/content/product_labels_test.csv"):
            embedding_model = load_embedding_model()
            product_data = load_product_data()
            
            if not product_data.empty:
                result_text, images = find_similar_products(entities, product_data, embedding_model)
                print(f"검색 결과:\n{result_text[:200]}...")
                print(f"결과 이미지 수: {len(images)}")
            else:
                print("상품 데이터가 비어있습니다.")
        else:
            print("상품 데이터 파일이 없습니다.")
    else:
        print("OpenAI API 키가 설정되지 않았습니다.")

# 테스트 실행 (선택사항)
# quick_test()

print("코드 블록 6 완료!")