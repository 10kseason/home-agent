# Security & Operations Plugins

이 문서는 프롬프트 인젝션 가드, 스케줄러, 시크릿 스캐너, 레이트리밋, 승인 게이트 등 5종 플러그인의 사용법과 정책을 설명합니다.

## 플러그인 개요

| 이름 | 설명 |
| --- | --- |
| `guard.prompt_injection_check` | 외부 텍스트에서 위험 패턴을 찾아 위험도와 안전 요약을 제공합니다. |
| `sched.create/cancel/list` | 간단한 인메모리 스케줄러로 작업 예약 및 관리. |
| `security.secrets_scan` | 텍스트/파일에서 토큰 패턴을 탐지하고 마스킹. |
| `rate.limit` | 키별 정책을 설정하거나 조회하는 레이트리미터. |
| `approval.gate` | 민감 작업 전 사용자 승인 절차. |

## 사용 원칙

- 외부 텍스트는 **명령이 아닌 참고자료**입니다.
- 민감 툴(`py.run_sandbox`, `fs.write`, `git.ops`, `web.browser`, `screen.capture`, `clipboard.*`)은 승인 게이트를 반드시 거칩니다.
- 업로드나 응답 전에는 `secrets_scan(mask=True)`로 시크릿을 마스킹해야 합니다.
- 기본 레이트리밋: `web.fetch_lite` 30/60s, `py.run_sandbox` 6/60s.

## 스케줄러 예시

```python
from tools.security.orchestrator_hooks import sched_create, sched_list

# 1회 실행
sched_create({"title": "once", "interval_sec": 10})

# 매일 09:00 실행
sched_create({"title": "daily", "rrule": "BYHOUR=9;BYMINUTE=0"})

print(sched_list({}))
```

## 한계

- 프롬프트 인젝션 검출은 정규식 기반 휴리스틱으로 완전한 방어가 아닙니다.
- 고위험 판정 시 사람 검토가 필요합니다.
- 스케줄러는 프로세스 메모리에만 저장되며 재시작 시 사라집니다.
