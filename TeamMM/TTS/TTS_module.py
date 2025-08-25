# 라이브러리
from pydub import AudioSegment
from transformers import pipeline
import torch
from openai import OpenAI
import os

# TTS 모듈
class TTSModule:
    def __init__(self, openai_api_key):\
        # OpenAI 클라이언트 설정
        self.client = OpenAI(api_key=openai_api_key)
        
        # Whisper 모델 로드
        self.model_name = "SungBeom/whisper-small-ko"
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        print("TTS 모듈 로딩 완료!")
    
    # m4a나 다른 오디오 형식을 wav로 변환
    def convert_audio_to_wav(self, audio_path):
        if audio_path.endswith('.wav'):
            return audio_path
        
        wav_path = audio_path.replace(os.path.splitext(audio_path)[1], ".wav")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            print(f"오디오 변환 실패: {e}")
            return None
    
    # 음성을 텍스트로 변환
    def speech_to_text(self, audio_path):
        try:
            # 필요시 wav로 변환
            wav_path = self.convert_audio_to_wav(audio_path)
            if wav_path is None:
                return "오디오 변환 실패"
            
            # Whisper로 텍스트 인식
            result = self.asr(wav_path)
            return result["text"]
        
        except Exception as e:
            print(f"음성 인식 실패: {e}")
            return f"음성 인식 오류: {str(e)}"
    
    # 텍스트에서 상품명과 특징 추출
    def extract_entities_with_features(self, text):
        
        PROMPT = '당신은 상품 개체 인식 전문가입니다.'
        
        instruction = f'''
        다음은 사용자가 말한 문장입니다.
        이 문장에서 장바구니에 담을 상품명과 주요 특징을 함께 포함해서,
        각 항목을 '상품명' 형태로 뽑아주세요. 여러 개는 쉼표로 구분된 리스트 형태로 간결하게 뽑아주세요.
        
        문장: "{text}"
        
        예시)
        입력: 오뚜기 카레 분말 하나랑, 햇반 작은 거 두 개 넣어줄래?
        출력: 오뚜기 카레 분말, 햇반 작은 거
        
        입력: 파란색 포카칩이랑 보라색깔 포키 하나 담아줘
        출력: 파란색 포카칩, 보라색깔 포키
        
        입력: 바나나 우유 하나랑 초코우유 두 개 담아줘
        출력: 바나나 우유, 초코우유
        '''
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": instruction}
                ],
                temperature=0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            entities = [e.strip() for e in result_text.split(",") if e.strip()]
            return entities
        
        except Exception as e:
            print(f"엔티티 추출 실패: {e}")
            return [text]  # 실패시 원본 텍스트 반환
    
    # 음성 파일을 처리해서 인식된 텍스트와 엔티티 반환
    def process_audio(self, audio_path):
        
        # 1. 음성 → 텍스트
        recognized_text = self.speech_to_text(audio_path)
        
        if "오류" in recognized_text or "실패" in recognized_text:
            return recognized_text, []
        
        # 2. 텍스트 → 엔티티 추출
        entities = self.extract_entities_with_features(recognized_text)
        
        return recognized_text, entities

# 실행
if __name__ == "__main__":
    # API 키 설정 
    OPENAI_API_KEY = ""
    
    # 모듈 초기화
    tts = TTSModule(OPENAI_API_KEY)
    
    # 테스트 파일
    test_file = "/home/work/XAI_ADV/Training/쿼리 샘플2.m4a"
    
    if os.path.exists(test_file):
        text, entities = tts.process_audio(test_file)
        print(f"인식된 텍스트: {text}")
        print(f"추출된 엔티티: {entities}")
    else:
        print("테스트 파일이 없습니다.")