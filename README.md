# Overlay LLM Orchestrator (Experimental)

> ⚠️ **본 프로젝트는 GPT, Claude, 기타 앱들을 활용해 만든 결과물입니다.**  
> ⚠️ **코드의 99%는 바이브 코딩(Vibe Coding)으로 작성되었습니다.**  
> ⚠️ 저는 단지 주도하여 이어붙이고 테스트했습니다. 완벽한 작품이 아닙니다.

---

## 🚀 설치 및 실행

압축 풀고 처음 사용자용 실행 및 설치.bat를 실행합니다. LM Studio가 설치되어 있어야 하며
OCR 폴더, Overlay 폴더, STT 폴더, Agent 폴더에서 Config 파일에 적힌
절대경로 위치를 여러분의 컴퓨터의 절대경로 위치로 바로 잡아주시길 바랍니다.

📦 필요 모델 목록
Qwen3 2507 4B
Qwen2.5 VL 7B
Qwen3 8B
LFM2-VL 1.6B
HyperClova SEED 1.5B
GPT-OSS-20B (선택적)
Qwen3 14B (선택적)
LM studio에서 각 모델을 받아야 합니다.

Whisper Tiny
Whisper Faster는 STT 실행시 설치됩니다. 받는데 시간이 걸릴 수 있습니다.

최소 필요사양

24GB 이상 램,

대략 50GB 이상의 공간.

16GB 이상의 VRAM. (CUDA 에서만 동작 확인됨)
---

## 🖥️ 개발 및 테스트 환경
- GPU: **RTX 4060 Ti 16GB**
- CPU: **AMD 7800X3D**
- RAM: **32GB**
- OS: **Windows 11**

해당 사양에서 실시간 STT(Whisper) 전사와 Overlay 기능이 구동 가능하며, OCR은 다소 시간이 걸리지만 사용 가능합니다.

---

## ⚙️ 동작 구조

### 기본 오케스트레이션
- **Qwen3 2507 4B**  
  → Overlay의 대부분 기능(툴 호출 담당)  

### OCR 파이프라인
- **LM Studio**에서 4B 종료 후:  
  - `Qwen2.5 VL 7B` : OCR 인식  
  - 종료 후 `Qwen3 8B` : 한국어 번역  
  - 결과 Overlay에 표시  
- **고속 모드**:  
  - `LFM2-VL 1.6B` : OCR  
  - `HyperClova SEED 1.5B` : 번역  

### STT 파이프라인
- `Whisper Tiny / Faster` : 음성 전사 (BBC 뉴스도 인식 가능)  
- `HyperClova SEED 1.5B` : 전사 결과물 한국어 번역  
- 결과는 Overlay 및 `VSRG-Ts-to-KR.py (STT)` 창에 표시, **화자 분리 지원**  

### 오케스트레이션 순환
- Overlay에 대화 전송 → LMS 설정에 따라 8B 종료 → 4B가 다시 툴 호출  
- 필요 시 OCR / STT 자동 불러오기 후 작업 완료 시 종료  

### 추가 기능
- 웹 검색 기능 포함 (현재 일부 버그 존재)  
- `/Pin14b`, `/Pin20b` 명령어로 **Qwen3 14B**, **GPT-OSS-20B** 모델 호출 가능  
- 대부분의 툴 콜링은 4B 수준에서 처리됨  
- 한국어 환경에서만 테스트 완료  

---

## 📦 필요 모델 목록
- **Qwen3 2507 4B**
- **Qwen2.5 VL 7B**
- **Qwen3 8B**
- **LFM2-VL 1.6B**
- **HyperClova SEED 1.5B**
- **GPT-OSS-20B** (선택적)
- **Qwen3 14B** (선택적)
- **Whisper Tiny**
- **Whisper Faster**

---

## 🔑 라이선스
- MIT License
- 자유롭게 개조 및 사용 가능  
- 단, **원본 GitHub 저장소 출처를 반드시 표기**해주세요  

---

## 🙋‍♂️ 마무리
이 프로젝트는 제가 AI와 협업하여 시행착오 끝에 만들어낸 실험적 결과물입니다.  
사양이 되는 분들은 직접 돌려보시고, 마음껏 개조해주시면 감사하겠습니다.  
