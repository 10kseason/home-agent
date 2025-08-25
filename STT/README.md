# VSRG Translator Pro

정교한 **시스템 출력 → STT(영/다국어) → 번역(한글) → 오버레이** 파이프라인입니다.  
WASAPI 루프백 + `soundcard` 폴백, WebRTC VAD + **RMS 기반 강제 세그먼트**, **로컬/서버 STT 선택**, LM Studio 번역을 지원합니다.

## 설치
```bash
pip install -r requirements.txt
```

## 실행
```bash
python app.py --list-devices   # 참고용, 장치 확인
python app.py                  # 기본: 자동 루프백 캡처 + 로컬 STT + 번역
```

## 핵심 설정 (`config.yaml`)
- `capture.mode`: `"auto"`(기본) → WASAPI (loopback) 시도, 실패 시 `soundcard`로 기본 스피커 루프백
- `capture.mode: "device"` + `device_index`로 **특정 장치 고정**
- `capture.mode: "app"` + `apps` 배열로 **특정 프로그램 세션만 캡처** (예: `Discord.exe`, `chrome.exe`)
- `stt.backend`: `"local"`(faster-whisper) 또는 `"server"`(내가 제공한 `stt_server.py` 같은 서버)
- `force.enable: true` → **VAD가 말을 못 잡아도 RMS 기준으로 강제 STT**

자주 바꾸는 값 예시:
```yaml
capture:
  mode: "device"
  device_index: 66           # 예: Input (VB-Audio Point)
stt:
  backend: "local"
  model: "small.en"
  compute_type: "auto"       # GPU면 float16, CPU면 int8
  language: "en"             # 자동감지면 null
force:
  rms_speech_threshold_dbfs: -42.0
  sustained_loud_ms: 2000
translate:
  enable: true
```

## OBS / VB-Audio 팁
- OBS **Monitoring Device**를 `Output (VB-Audio Point)`로, 대상 소스는 **Monitor Only**  
- 앱은 `device_index`를 **`Input (VB-Audio Point)`**로 지정
- Windows **개인정보 보호 → 마이크 권한**은 켜둬야 가상 녹음장치 열림 (마이크 물리 입력을 쓰진 않음)

## 디버깅
- 콘솔:
  - `[INFO] Using sounddevice ...` 또는 `soundcard loopback engine` 
  - `segment ready: ... rms=... dBFS` 또는 `[force] sustained/loud->quiet`
- `debug.write_wav_segments: true`면 각 세그먼트를 `./segments/`에 저장

문제 있으면 콘솔 로그 몇 줄과 `--list-devices` 출력 주면 더 조여 드립니다.
