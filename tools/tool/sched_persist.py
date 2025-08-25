# tools/tool/sched_persist.py
from __future__ import annotations
import subprocess, shlex, os, sys

def sched_persist(args):
    title = args["title"]
    delete = bool(args.get("delete", False))
    if delete:
        cmd = f'schtasks /Delete /TN "{title}" /F'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {"deleted": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
    time_s = args.get("time")
    every  = args.get("every")     # DAILY|HOURLY|MINUTE:n
    cmdline = args.get("cmd")
    if not cmdline:
        # 기본: 현재 파이썬으로 README 출력 같은 무해한 명령
        py = sys.executable.replace("\\","\\\\")
        cmdline = f'"{py}" -c "print(\\"hello from sched.persist\\")"'
    if every and every.upper().startswith("MINUTE:"):
        n = every.split(":",1)[1]
        sch = f'schtasks /Create /TN "{title}" /SC minute /MO {n} /TR {cmdline} /F'
    elif every and every.upper()=="HOURLY":
        sch = f'schtasks /Create /TN "{title}" /SC hourly /TR {cmdline} /F'
    else:
        t = time_s or "12:00"
        sch = f'schtasks /Create /TN "{title}" /SC daily /ST {t} /TR {cmdline} /F'
    r = subprocess.run(sch, shell=True, capture_output=True, text=True)
    return {"created": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}

TOOL_HANDLERS = {"sched.persist": sched_persist}
