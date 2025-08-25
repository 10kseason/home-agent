# Luna Overlay (Semi-Transparent Orchestrator)

## Install
```
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python overlay_app.py
```
Edit `config.yaml` for your endpoints and model.

## JSON response format (LLM must output)
{"say":"...", "tool_calls":[{"name":"ocr.start","args":{"hint":"subtitle"}}, {"name":"agent.event","args":{"type":"note","payload":{"k":"v"}}}]}
