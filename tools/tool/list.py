import json
import sys
from .registry import ToolRegistry
from .base import ToolContext

def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "tools/config/tools.yaml"
    reg = ToolRegistry.from_yaml(yaml_path, ctx=ToolContext())
    print(json.dumps(reg.list_tools(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
