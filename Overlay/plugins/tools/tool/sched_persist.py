
from __future__ import annotations
import subprocess, sys, json

def sched_persist(args):
    title = args["title"]
    delete = bool(args.get("delete", False))
    if delete:
        cmd = f'schtasks /Delete /TN "{title}" /F'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {"deleted": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
    time_s = args.get("time") or "12:00"
    every  = (args.get("every") or "").upper()    # DAILY|HOURLY|MINUTE:n
    cmdline = args.get("cmd") or 'cmd /c echo hello-from-sched.persist'

    if every.startswith("MINUTE:"):
        n = every.split(":",1)[1]
        sch = f'schtasks /Create /TN "{title}" /SC minute /MO {n} /TR {cmdline} /F'
    elif every == "HOURLY":
        sch = f'schtasks /Create /TN "{title}" /SC hourly /TR {cmdline} /F'
    else:
        sch = f'schtasks /Create /TN "{title}" /SC daily /ST {time_s} /TR {cmdline} /F'

    r = subprocess.run(sch, shell=True, capture_output=True, text=True)
    return {"created": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}

TOOL_HANDLERS = {"sched.persist": sched_persist}

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    args = payload.get("args", payload)
    try:
        print(json.dumps(sched_persist(args), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
