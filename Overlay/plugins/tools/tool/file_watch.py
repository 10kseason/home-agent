import os
import time
import glob
import json
import threading
from typing import Dict, Any, Set
from .base import ToolPlugin, ToolContext

class FileWatch(ToolPlugin):
    name = "file.watch"
    description = "폴더를 폴링 방식으로 감시하고 이벤트를 agent_output/watch_events.jsonl에 적재"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string", "default": "*.*"},
            "on_event": {"type": "string", "enum": ["create", "modify", "delete"], "default": "create"},
            "interval_sec": {"type": "integer", "default": 2}
        },
        "required": ["path"],
        "additionalProperties": False
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        path = args["path"]
        pattern = args.get("pattern", "*.*")
        on_event = args.get("on_event", "create")
        interval = int(args.get("interval_sec", 2))

        out_dir = os.path.join("agent_output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "watch_events.jsonl")

        def snapshot() -> Dict[str, float]:
            result = {}
            for p in glob.glob(os.path.join(path, pattern), recursive=True):
                if os.path.isfile(p):
                    try:
                        st = os.stat(p)
                        result[p] = st.st_mtime
                    except Exception:
                        pass
            return result

        def loop():
            prev = snapshot()
            created_seen: Set[str] = set(prev.keys())
            while True:
                time.sleep(interval)
                cur = snapshot()

                created = [p for p in cur.keys() if p not in created_seen]
                deleted = [p for p in prev.keys() if p not in cur]
                modified = [p for p in cur.keys() if p in prev and cur[p] != prev[p]]

                events = []
                if on_event == "create":
                    events = created
                    created_seen.update(created)
                elif on_event == "modify":
                    events = modified
                elif on_event == "delete":
                    events = deleted

                for p in events:
                    rec = {"event": on_event, "path": p, "ts": time.time()}
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ctx.event_bus("file.watch", rec)

                prev = cur

        t = threading.Thread(target=loop, name="file.watch", daemon=True)
        t.start()
        return self.ok(message="watcher started", path=path, pattern=pattern, on_event=on_event, interval=interval)
