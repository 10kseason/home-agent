# overlay_app.py의 탭을 전부 스페이스 4칸으로 치환
$path="D:\jip\home-agent\Overlay\overlay_app.py"; 
(Get-Content $path -Raw).Replace("`t","    ") | Set-Content $path -Encoding UTF8
