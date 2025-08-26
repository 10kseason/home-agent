
"""
Overlay Plugin Bridge
---------------------
If your Overlay orchestrator exposes a method:
    register_tool(name: str, schema: dict, handler: callable)
then you can do:

    from tools.tool import register as register_tools
    register_tools(overlay)

Alternatively, import ToolRegistry to execute by name with safety checks:

    from tools.tool.runtime_registry import ToolRegistry
    reg = ToolRegistry.from_yaml("tools/config/tools.yaml")
    res = reg.execute("web.fetch_lite", {"url": "https://example.com"})

"""
