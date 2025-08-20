# KMU-Smart-Parking-Monitor

## 🏎 배경 및 주제
- 국민대학교 지하주차장 이용 현황
  - 부정확한 층별 주차 가능 대수 안내
  - 천장 주차 가능 안내 조명 부재
  - 이로 인한 주차 스트레스 유발
- 솔루션 : CCTV 기반 객체 탐지를 통해 주차 공간 점유 여부 실시간 파악 및 UI 구현


## 💁 팀원 
- 총 5인 : 송승원, 오서영, 이수빈, 조현식(팀장), 홍예진


## 📅 진행기간
- 2025.07 ~ 2025.08 (2개월)

## 💻 Monitor
<img width="800" height="576" alt="monitor1" src="https://github.com/user-attachments/assets/67d8c4bd-e800-4924-b278-1b631d85682f" />
<img width="800" height="610" alt="monitor2" src="https://github.com/user-attachments/assets/e910de20-776b-4c33-890b-e750e02df07a" />


## 🚀 데이터셋 & 모델 
- 데이터 촬영 장소 : 국민대학교 지하주차장
- 촬영 각도 : CCTV 촬영 각도에 맞춰 촬영
- 총 학습용 이미지 수 : 166장
<img width="900" height="270" alt="data" src="https://github.com/user-attachments/assets/8543883a-86ea-4a63-bafa-ca56a192131b" />

- labellmg 툴 사용하여 직접 라벨링 진행
- 차량이 존재하는 경우 1, 빈칸일 경우 0으로 라벨링
- 라벨 형식 : YOLO (class_id, x_center , y_center, width, height)
<img width="900" height="332" alt="image" src="https://github.com/user-attachments/assets/721454a7-4ed0-4a5f-ac32-a02dad36b268" />

- 학습 모델 : YOLOv11 Nano (50 epoch)
- 모델 가중치 : https://drive.google.com/drive/folders/1T_5ctxvtpUOLLudylMz7tprvQpGQeUcw?usp=sharing
<img width="900" height="318" alt="image" src="https://github.com/user-attachments/assets/f7b0c7b3-f340-4613-a566-62063e921d07" />


## 🛠️ 설치 및 실행 방법
1. **Clone the repository**
   ```bash
   git clone https://github.com/hsjo827/KMU-Smart-Parking-Monitor.git
   cd KMU-Smart-Parking-Monitor
   ```
   
2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run**
   ```bash
   # 주차장 도면 & 적용 영상에 맞춰 parking_app/config.py에서 좌표 수정 필요 
   python main.py
   ```

## 👏 Reference
- [2025 ADV TOY PROJECT 최종 발표.pdf](https://github.com/user-attachments/files/21849986/2025.ADV.TOY.PROJECT.pdf)


## ⭐ Citation
```bash
@article{khanam2024yolov11,
  title   = {YOLOv11: An Overview of the Key Architectural Enhancements},
  author  = {Rahima Khanam, Muhammad Hussain},
  journal = {arXiv preprint arXiv:2410.17725},
  year    = {2024}
}
```

