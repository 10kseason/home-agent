# Luna Local Agent (Autonomous Orchestrator)

이 프로젝트는 네가 이미 가지고 있는 도구들(STT 번역, OCR→한국어, 디스코드 요약봇 등)을 **하나의 이벤트 중심 에이전트**로 묶어
"팔 붙이고 같이 움직이는" 로컬 AI를 만들기 위한 골격입니다.  
윈도우에서 실행 가능한 단일 EXE로 묶을 수 있게 설계되어 있습니다.

## 구성 요약
- **FastAPI 로컬 서버(127.0.0.1:8765)**: 외부/내부 툴이 이벤트를 POST `/event`로 던지면 에이전트가 구독한 플러그인이 처리
- **Event Bus**: 우선순위/중복제거/레이트리밋/규칙 적용
- **Plugin System**: STT, OCR, 번역, 디스코드 요약 등 독립 모듈로 확장
- **Autonomy Policy**: 자동/반자동/질문-후-실행의 레벨 설정
- **Sinks**: 결과를 오버레이·파일·토스트 알림 등으로 출력 (샘플: 로그/토스트)

> 기본값은 **완전 로컬**을 가정합니다. 번역/요약을 LM Studio 또는 Ollama(OpenAI 호환 엔드포인트)로 보내는 샘플이 포함됩니다.
> 클라우드 API를 쓰고 싶다면 `config.yaml`에서 엔드포인트와 키를 채우세요.

---




python -m agent.main

## 빠른 시작
```bash
# 1) 가상환경 (선택)
python -m venv .venv && .venv\Scripts\activate  # Windows PowerShell

# 2) 설치
pip install -r requirements.txt

# 3) 실행
python -m agent.main
```

서버는 기본적으로 `http://127.0.0.1:8765`에서 뜹니다.

### 이벤트 보내기 예시
```bash
# OCR 결과를 보냈다고 가정
curl -X POST http://127.0.0.1:8765/event ^
  -H "Content-Type: application/json" ^
  -d "{\"type\":\"ocr.text\",\"payload\":{\"text\":\"Hello world\",\"lang\":\"en\"},\"priority\":5}"
```

### 네 기존 툴과 연결
- **STT 번역 오버레이**: STT 툴의 새 자막이 나올 때마다 `/event`로 `stt.text` 이벤트를 POST합니다.
- **OCR→한국어**: 화면 캡처 후 텍스트가 나오면 `ocr.text` 이벤트를 POST합니다.
- **디스코드 요약봇**: 새 알림 모아서 5분마다 `discord.batch` 이벤트로 POST합니다.

각 툴에서 HTTP POST 한 줄만 추가하면 에이전트가 플러그인으로 후처리를 이어갑니다.

---

## 설정
`agent/config.yaml`:
```yaml
server:
  host: "127.0.0.1"
  port: 8765

policy:
  autonomy_level: "auto"   # auto | confirm | hybrid
  batch_seconds: 300       # 동일 소스 이벤트 묶음 주기(예: 디스코드)

translate:
  provider: "lmstudio"     # lmstudio | ollama | openai
  endpoint: "http://127.0.0.1:1234/v1"  # LM Studio 기본
  model: "qwen2.5-7b-instruct"
  api_key: ""              # (openai 등 클라우드 쓸때만)

sinks:
  toast: true
  log_file: "agent_output.log"

limits:
  dedup_window_seconds: 3600
  max_queue_size: 1000
```

---

## 플러그인
- `agent/plugins/translator_plugin.py`: `*.text` 이벤트를 감지해 한국어/영어 자동 번역
- `agent/plugins/ocr_plugin.py`: OCR 텍스트 후처리(줄나눔/노이즈 제거/언어감지)
- `agent/plugins/stt_plugin.py`: STT 자막 후처리(중복/반복 제거)
- `agent/plugins/discord_plugin.py`: 디스코드 알림 배치 요약

새 플러그인 추가: `agent/plugins/my_plugin.py` 파일을 만들고 `BasePlugin` 상속 + `handles` 목록을 채우면 자동 등록됩니다.

---

## EXE로 패키징
```bash
pip install pyinstaller
pyinstaller -F -n LunaLocalAgent agent/main.py
# dist/LunaLocalAgent.exe 생성
```

> FastAPI/uvicorn 사용 시 -F 단일 바이너리 모드는 용량이 커질 수 있습니다. 필요시 `--add-data` 등 옵션 조정.

---

## 보안/프라이버시
- 서버는 기본 `127.0.0.1`에만 바인딩되어 외부 접근 불가.
- 이벤트는 **네가 보낼 때만** 처리하며, 에이전트가 임의로 PC를 스캔하거나 감시하지 않습니다.
- 클라우드 번역을 켜면 해당 텍스트만 엔드포인트로 전송됩니다.

---

## 샘플 워크플로우
1) 디스코드 알림 여러개 → 5분마다 `discord.batch` → 요약 → 한국어 변환 → 토스트 알림 단 1개.  
2) 영어 영상 시청 → STT `stt.text` 이벤트 → 번역 → 오버레이/토스트.  
3) 화면 캡처 OCR → `ocr.text` 이벤트 → 언어 감지 후 한국어 번역 → 클립보드 복사 + 토스트.  

---

## License
MIT
