목표/UX: “디스코드 번역 해줘”가 들어오면 → 번역 → 클립보드 복사 → Toast+미니창 알림 자동. 붙여넣기는 사용자가 디스코드 앱에서 수동 수행.

방향 규칙 (전부 LLM만 사용)

KO → X(EN/JA/ZH …): LLM 체인 사용
1순위 qwen/qwen3-4b-2507 → 실패/부족 시 hyperclovax-seed-text-instruct-1.5b

X → KO: LLM만 사용(동일 체인: Qwen4B → HyperCLOVA 1.5B)

의도/언어 인식

트리거: “번역해줘/디스코드 번역/영어(일본어/중국어)로 …”

목표 언어 미지정 시 기본 EN.

한글 포함 여부로 원문=KO/비KO 판정(간단 휴리스틱).

VRAM 정책 (LLM 전용)

최대 동시 상주 모델: 2개(번역 1 + 기타 1)

소프트 상한: 총 VRAM의 90%

의사결정 순서

후보 모델 로드 전, 현 사용량 + 후보 예상 VRAM ≤ 상한이면 로드

초과 시:

언로드: 가장 크고 최근 사용 빈도 낮은 모델부터 제거(LRU, 쿨다운 30–60s)

강등: qwen/qwen3-4b-2507 → hyperclovax-seed-text-instruct-1.5b

그래도 초과/실패 시: 대기(큐잉) 또는 CPU/외부 API 임시 폴백(옵션)

큐잉/스로틀: 번역 요청 큐 최대 3건, 대기 2s↑면 경량 모델 우선

참고용 예상치(환경에 따라 조정):
Qwen3 4B(INT4) ≈ 4.5GB / HyperCLOVA-X Seed 1.5B ≈ 2.2GB

클립보드 & 알림

복사 대상

기본: 모바일 클립보드 (옵션: Windows 클립보드)

설정 clipboard.device = android | windows

알림

Toast(≈2.5s): “[영어] 번역이 클립보드에 복사되었어요.”

미니창(10s 자동 축소): 미리보기(앞 80자), “디스코드 열기/다시 번역/닫기”

설정 스니펫 (예시)
translation:
  default_target: en
  engine: llm                         # 전 방향 LLM만 사용
  llm_endpoint: http://127.0.0.1:1234/v1
  llm_model_primary:  "qwen/qwen3-4b-2507"
  llm_model_fallback: "hyperclovax-seed-text-instruct-1.5b"

clipboard:
  device: android                     # android | windows

ui:
  toast_duration_ms: 2500
  mini_sheet_timeout_ms: 10000

scheduler:
  vram_soft_cap_ratio: 0.90
  max_loaded_models: 2
  fallbacks:
    - { from: "qwen/qwen3-4b-2507", to: "hyperclovax-seed-text-instruct-1.5b" }

models:
  metadata:
    qwen/qwen3-4b-2507:                  { vram_mib_est: 4500 }
    hyperclovax-seed-text-instruct-1.5b: { vram_mib_est: 2200 }

처리 흐름 (9단계)

문장 수신(STT final/사용자 입력)

트리거/목표 언어 판정(미지정→EN)

원문 KO/비KO 판정(한글 포함 여부)

LLM 엔진 선택(Qwen4B → HyperCLOVA 1.5B)

VRAM 정책 적용(언로드/강등/대기)

번역 실행

클립보드 복사

Toast + 미니창 알림

로그(방향/엔진/지연/VRAM조치)