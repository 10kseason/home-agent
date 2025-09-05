#!/usr/bin/env python3
"""
오디오 디바이스 테스트 스크립트
"""
import sounddevice as sd

def list_audio_devices():
    """사용 가능한 오디오 디바이스 목록 출력"""
    print("=== 사용 가능한 오디오 디바이스 ===")
    devices = sd.query_devices()
    
    input_devices = []
    for i, device in enumerate(devices):
        channels_in = device.get('max_input_channels', 0)
        channels_out = device.get('max_output_channels', 0)
        name = device.get('name', 'Unknown')
        
        device_type = []
        if channels_in > 0:
            device_type.append(f"IN({channels_in}ch)")
            input_devices.append(i)
        if channels_out > 0:
            device_type.append(f"OUT({channels_out}ch)")
            
        type_str = " ".join(device_type) if device_type else "NO I/O"
        print(f"  [{i:2d}] {name} - {type_str}")
    
    print(f"\n=== 기본 디바이스 ===")
    try:
        default = sd.default.device
        print(f"Default device: {default}")
        if isinstance(default, (tuple, list)) and len(default) >= 2:
            print(f"  Input:  {default[0]}")
            print(f"  Output: {default[1]}")
    except Exception as e:
        print(f"기본 디바이스 정보 가져오기 실패: {e}")
    
    print(f"\n=== 입력 가능한 디바이스 ({len(input_devices)}개) ===")
    for i in input_devices:
        device = devices[i]
        print(f"  [{i:2d}] {device.get('name', 'Unknown')} ({device.get('max_input_channels', 0)}ch)")
    
    return input_devices

def test_device(device_id):
    """특정 디바이스로 짧은 녹음 테스트"""
    print(f"\n=== 디바이스 {device_id} 테스트 ===")
    try:
        import numpy as np
        duration = 1  # 1초
        samplerate = 16000
        
        print(f"디바이스 {device_id}로 {duration}초간 녹음 테스트...")
        audio_data = sd.rec(
            int(duration * samplerate), 
            samplerate=samplerate, 
            channels=1, 
            device=device_id,
            dtype='float32'
        )
        sd.wait()  # 녹음 완료까지 대기
        
        # 오디오 레벨 확인
        rms = np.sqrt(np.mean(audio_data**2))
        max_val = np.max(np.abs(audio_data))
        print(f"✅ 성공! RMS: {rms:.6f}, Max: {max_val:.6f}")
        
        if max_val < 0.001:
            print("⚠️  경고: 오디오 신호가 너무 낮습니다. 마이크가 음소거되었거나 작동하지 않을 수 있습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False

if __name__ == "__main__":
    # 디바이스 목록 출력
    input_devices = list_audio_devices()
    
    # 기본 입력 디바이스 테스트
    if input_devices:
        try:
            default_input = sd.default.device[0] if isinstance(sd.default.device, (tuple, list)) else sd.default.device
            if default_input in input_devices:
                test_device(default_input)
            else:
                # 첫 번째 입력 디바이스 테스트
                test_device(input_devices[0])
        except:
            if input_devices:
                test_device(input_devices[0])
    else:
        print("❌ 입력 디바이스를 찾을 수 없습니다!")