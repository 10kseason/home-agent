# Overlay LLM Orchestrator (Experimental)

> ⚠️ **본 프로젝트는 GPT, Claude, 기타 앱들을 활용해 만든 결과물입니다.**  
> ⚠️ **코드의 99%는 바이브 코딩(Vibe Coding)으로 작성되었습니다.**  
> ⚠️ 저는 단지 주도하여 이어붙이고 테스트했습니다. 완벽한 작품이 아닙니다.

---

## 🚀 설치 및 실행

압축을 풀고 `처음 사용자용 실행 및 설치.bat`를 실행합니다. LM Studio가 설치되어 있어야 하며
**OCR/`config.json`, STT/`config.yaml`, Overlay/`config.yaml`, agent/`config.yaml` 등 각 폴더의 설정 파일에 기재된 모델·파일 경로는 PC마다 모두 다릅니다. 반드시 자신의 절대경로로 수정하세요.**
**하드웨어 사양에 따라 모델 종류나 옵션을 바꿔 써도 무방합니다.**

📦 필요 모델 목록
opengvlab_internvl3_5-8b
opengvlab_internvl3_5-4b (고속 OCR)
Qwen3 2507 4B
Qwen3 8B
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
  - `opengvlab_internvl3_5-8b` : OCR 인식
  - 종료 후 `Qwen3 8B` : 한국어 번역
  - 결과 Overlay에 표시
- **고속 모드**:
  - `opengvlab_internvl3_5-4b` : OCR
  - `kakaocorp_kanana-1.5-2.1b-instruct-2505` : 번역

### STT 파이프라인
- `Whisper Tiny / Faster` : 음성 전사 (BBC 뉴스도 인식 가능)
- `opengvlab_internvl3_5-8b` : 전사 결과물 한국어 번역
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
- **opengvlab_internvl3_5-8b**
- **Qwen3 8B**
- **GPT-OSS-20B** (선택적)
- **Qwen3 14B** (선택적)
- **opengvlab_internvl3_5-4b** (고속 OCR)
- **Whisper Tiny**
- **Whisper Faster**

---

## 🔑 라이선스
- GPL-3.0 License
- 이 저장소의 코드는 GPLv3로 배포됩니다.
- 이 프로젝트는 모델(가중치)을 포함하지 않습니다. 사용자는 각 모델의 라이선스 조건을 확인 후 직접 다운로드해야 합니다.
- "오픈소스" 표기는 코드에 한정되며, 모델은 각 저작권자/라이선스를 따릅니다.
- 단, **원본 GitHub 저장소 출처를 반드시 표기**해주세요

### 📚 의존 라이브러리 라이선스
아래는 `requirements.txt`에 명시된 주요 라이브러리와 그 라이선스입니다. 대부분은 GPLv3와 호환되지만, **PyQt5는 GPLv3**입니다.

| 라이브러리 | 라이선스 |
|------------|----------|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic | MIT |
| httpx | BSD-3-Clause |
| requests | Apache-2.0 |
| PyYAML | MIT |
| loguru | MIT |
| PyQt5 | GPLv3 |
| PySide6 | LGPL-3.0 |
| pillow | HPND (PIL) |
| mss | MIT |
| keyboard | MIT |
| sounddevice | MIT |
| soundfile | BSD-3-Clause |
| soundcard | BSD-3-Clause |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| webrtcvad | MIT |
| faster-whisper | MIT |
| librosa | ISC |
| scikit-learn | BSD-3-Clause |
| pycaw | MIT |
| comtypes | MIT |
| duckduckgo-search | MIT |
| beautifulsoup4 | MIT |
| lxml | BSD-3-Clause |

이외 윈도우 전용 알림 라이브러리(win10toast, winotify)는 모두 MIT 라이선스로 배포됩니다.

**GPLv3 준수:** PyQt5는 GPLv3이므로 프로젝트 전체는 GPL 조항을 따릅니다. 나머지 라이브러리는 GPLv3와 호환되어 자유롭게 사용할 수 있습니다.

---

## 🙋‍♂️ 마무리
이 프로젝트는 제가 AI와 협업하여 시행착오 끝에 만들어낸 실험적 결과물입니다.  
사양이 되는 분들은 직접 돌려보시고, 마음껏 개조해주시면 감사하겠습니다.  
