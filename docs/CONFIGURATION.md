# Configuration Guide / 구성 가이드

## Overview / 개요
The `config.yaml` file centralizes settings for the agent, overlay, OCR, and STT modules. It also defines shared budgets, translation rules, and privacy profiles.
`config.yaml` 파일은 에이전트, 오버레이, OCR, STT 모듈의 설정을 한 곳에 모아 관리하며, 공용 예산과 번역 규칙, 프라이버시 프로필을 정의합니다.

## Sections / 섹션
### Agent / 에이전트
Controls server host/port, policy automation level, translation backend, output sinks, and queue limits.
서버 호스트/포트, 정책 자동화 수준, 번역 백엔드, 출력 싱크, 큐 제한 등을 제어합니다.

### Overlay / 오버레이
Configures visual style, launch command, LLM endpoints, proxy/UI behavior, and built-in tools.
시각적 스타일, 실행 명령, LLM 엔드포인트, 프록시/ UI 동작, 기본 도구를 설정합니다.

### OCR / OCR
Specifies OCR and translation models, hotkeys, clipboard behavior, and confidence thresholds.
OCR 및 번역 모델, 단축키, 클립보드 동작, 신뢰도 임계값을 지정합니다.

### STT / STT
Sets audio capture devices, VAD and forced speech parameters, STT backend, translation, and preview UI.
오디오 캡처 장치, VAD 및 강제 음성 매개변수, STT 백엔드, 번역, 미리보기 UI를 설정합니다.

### Voice & Budgets / 음성 및 예산
Lists voice-command synonyms, token budgets for thinking/answers, ranking weights, and privacy rules.
음성 명령 동의어, 사고/응답 토큰 예산, 순위 가중치, 프라이버시 규칙을 나열합니다.

For additional details, see inline comments in `config.yaml` which are now provided in both English and Korean.
자세한 내용은 영어와 한국어로 제공되는 `config.yaml`의 인라인 주석을 참고하십시오.
