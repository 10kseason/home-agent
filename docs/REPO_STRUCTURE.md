# Repository Structure

본 문서는 저장소의 디렉터리와 파일을 한눈에 이해할 수 있도록 설명합니다. 각 항목에는 파일 역할과 주요 태그를 기재하여 사람이든 AI든 빠르게 맥락을 파악할 수 있도록 구성했습니다.

## 최상위 파일
- `AGENTS.md` — [Markdown, Instructions] 저장소 사용 시 실행해야 할 명령과 규칙을 정의합니다.
- `LICENSE` — [GPLv3, License] 저장소 전체의 기본 라이선스가 되는 GPLv3 문서입니다.
- `README.md` — [Markdown, Docs] 프로젝트 개요와 설치 방법, 모델 요구사항 등을 설명합니다.
- `requirements.txt` — [Config, Dependencies] 필요한 파이썬 라이브러리를 나열합니다.
- `overlay_plugin_bridge.py` — [Python, Bridge] Overlay에서 도구를 등록하고 실행하는 방법을 안내하는 예시 코드입니다.
- `처음 사용자용 실행 및 설치.bat` — [Batch, Setup] 윈도우 사용자를 위한 초기 설치/실행 스크립트입니다.

## 디렉터리 개요

### `agent/`
경량 이벤트 기반 에이전트 서버 코드.
- `__init__.py` — [Python] 플러그인 로더를 제공하고 하위 모듈을 초기화합니다.
- `Types.py` — [Python, Types] 이벤트 및 페이로드 형식 정의.
- `config.yaml` — [YAML, Config] 서버 설정과 한계값을 정의합니다.
- `context.py` — [Python] 설정을 읽어 `EventBus`와 정책 객체를 준비합니다.
- `event_bus.py` — [Python, Messaging] 비동기 이벤트 큐와 중복 제거 로직을 구현합니다.
- `main.py` — [Python, Entrypoint] 설정을 로드하고 플러그인을 등록한 뒤 Uvicorn 서버를 실행합니다.
- `policy.py` — [Python] 자율성 수준을 정의하는 간단한 정책 객체입니다.
- `schemas.py` — [Python, Pydantic] 이벤트 데이터 모델.
- `server.py` — [Python, FastAPI] HTTP 엔드포인트를 제공하여 이벤트를 수신합니다.
- `sinks.py` — [Python] 이벤트를 외부 서비스로 내보내는 싱크 모듈 모음.
- `plugins/` — [Python, Plugins] 에이전트가 사용할 플러그인.
  - `__init__.py` — 플러그인 자동 로더.
  - `discord_plugin.py` — Discord Webhook으로 이벤트 전달.
  - `ocr_plugin.py` — OCR 결과 텍스트 정리 후 재전송.
  - `overlay_sink.py` — 이벤트를 Overlay UI에 전달.
  - `stt_plugin.py` — STT 결과 후처리 및 재전송.
  - `translator_plugin.py` — 번역 모델 호출을 담당.
  - `백업/overlay_sink_plugin.py` — [Legacy] 과거 버전의 Overlay 싱크 플러그인 백업본.
- `tools/send_event_example.py` — [Python, Example] 이벤트 전송 샘플 스크립트.

### `Overlay/`
GUI Overlay 애플리케이션과 해당 에이전트 복사본을 포함.
- `__init__.py` — 패키지 초기화.
- `LICENSE` — Overlay 모듈의 GPLv3 라이선스 사본.
- `README.md` — Overlay 모듈 사용법.
- `config.yaml` / `config_process_example.yaml` — Overlay 설정 예제.
- `overlay_app.py` — 기본 Overlay GUI 애플리케이션.
- `overlay_app_enhanced.py` — 확장 기능이 추가된 Overlay 버전.
- `overlay_plugin_system.py` — 플러그인 등록과 실행을 위한 시스템 코드.
- `tool_memory.txt` — [Text] 도구 사용과 관련한 메모 저장 파일.
- `agent/` — Overlay에 내장된 에이전트 코드(상위 `agent/`와 동일 구조).
- `plugins/` — Overlay에서 사용 가능한 플러그인 모음.
  - `open_url.py` — URL을 브라우저로 여는 간단한 플러그인.
  - `web_search.py` — DuckDuckGo 검색을 수행.
  - `webtools.py` — 웹 관련 툴 집합 래퍼.
  - `test/` — 테스트용 스크립트 모음.
  - `tools/` — Overlay용 툴 시스템(루트 `tools/`와 동일 구조).
    - `__init__.py` — 플러그인 시스템 초기화.
    - `config/tools.yaml` — 사용 가능한 툴 정의.
    - `overlay_plugin.py` — Overlay에 툴 등록 헬퍼.
    - `security/` — 툴 보안 훅과 플러그인.
      - `orchestrator_hooks.py` — 실행 전/후 훅 정의.
      - `security.py` — 보안 관련 유틸리티.
      - `security_plugins.py` — 샌드박싱/필터링 플러그인.
    - `tool/` — 실제 툴 구현체.
      - `agent_list_tools.py` — 실행 중인 에이전트 목록을 조회.
      - `base.py` — 모든 툴의 추상 기본 클래스.
      - `echo.py` — 입력을 그대로 반환.
      - `file_watch.py` — 파일 변경 감시.
      - `list.py` — 등록된 툴 목록을 반환.
      - `mail_imap_list.py` — IMAP 서버에서 메일 목록 확인.
      - `ocr.py` — OCR 이벤트 트리거.
      - `registry.py` — 레지스트리에 툴을 등록/관리.
      - `runtime_registry.py` — 런타임 툴 등록소.
      - `sched_persist.py` — 스케줄러 상태 영속화.
      - `security_tools.py` — 보안 관련 툴 모음.
      - `stt.py` — STT 이벤트 트리거.
      - `web_fetch_lite.py` — 경량 웹 요청 수행.
    - `tool.py` — 단일 툴 실행을 위한 러닝타임 헬퍼.
    - `tools/` — 툴 묶음을 제공하는 레이어.

### `OCR/`
- `config.json` — OCR 도구의 기본 설정.
- `main.py` — [Python, Qt] 스크린샷 영역을 캡처하고 OCR→번역을 수행하는 GUI 프로그램.

### `STT/`
음성 인식 및 번역 도구.
- `README.md` — 사용 안내.
- `config.yaml` — STT 설정.
- `VSRG-Ts-to-kr.py` — 텍스트 파일에서 영어 자막을 한국어로 변환.
- `debug.py` — STT 디버깅용 스크립트.
- `diarizer.py` — 화자 분리 기능 구현.

### `tests/`
- `test_security_plugins.py` — 보안 플러그인의 이벤트 필터링 동작 테스트.
- `test_stt_plugin.py` — STT 플러그인이 이벤트를 전달하는지 검증.

### `tools/`
공유 툴 시스템(Overlay/plugins/tools와 동일 구조).
- `__init__.py` — 패키지 초기화.
- `config/tools.yaml` — 사용 가능한 툴 목록.
- `overlay_plugin.py` — Overlay 측에 툴을 등록하기 위한 헬퍼.
- `security/`
  - `orchestrator_hooks.py` — 실행 전/후 훅 정의.
  - `security.py` — 보안 유틸리티.
  - `security_plugins.py` — 샌드박싱/필터링 플러그인.
- `tool/`
  - `agent_list_tools.py` — 실행 중인 에이전트 목록을 조회.
  - `base.py` — 모든 툴의 추상 기본 클래스.
  - `echo.py` — 입력을 그대로 반환.
  - `file_watch.py` — 파일 변경 감시.
  - `list.py` — 등록된 툴 목록을 반환.
  - `mail_imap_list.py` — IMAP 서버에서 메일 목록 확인.
  - `ocr.py` — OCR 이벤트 트리거.
  - `registry.py` — 레지스트리 기반 툴 관리.
  - `runtime_registry.py` — 런타임 툴 등록소.
  - `sched_persist.py` — 스케줄러 상태 영속화.
  - `security_tools.py` — 보안 관련 툴 모음.
  - `stt.py` — STT 이벤트 트리거.
  - `web_fetch_lite.py` — 경량 웹 요청 수행.
- `tool.py` — 단일 툴 실행을 위한 러닝타임 헬퍼.
- `tools/`
  - `agent_list_tools.py` — 에이전트 목록을 반환하는 래퍼.
  - `list.py` — 등록된 툴 나열.
  - `runtime_registry.py` — 런타임 툴 등록소 래퍼.
  - `sched_persist.py` — 스케줄러 상태 유지 래퍼.
  - `security_tools.py` — 보안 관련 툴 래퍼.
  - `web_fetch_lite.py` — 웹 요청 래퍼.

### 기타 디렉터리
- `OCR/__pycache__`, `STT/__pycache__`, `agent/__pycache__` 등 `__pycache__` 폴더는 각 파이썬 모듈의 컴파일된 바이트코드(`*.pyc`)를 보관합니다.
- 숨김 파일 및 `.git` 디렉터리는 Git 버전 관리 메타데이터입니다.

본 구조 문서는 모든 디렉터리와 주요 파일을 포괄적으로 설명하여, 향후 유지보수나 확장 시 빠르게 이해할 수 있도록 돕습니다.
