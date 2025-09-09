import importlib, pkgutil
from typing import List, Type

class BasePlugin:
    name: str = "base"
    handles: List[str] = []  # event type prefixes this plugin can handle

    def __init__(self, ctx=None):
        # ctx is optional to ease isolated unit tests. In production it is
        # provided by the agent and exposes bus, policy, config, sinks, etc.
        self.ctx = ctx

    async def handle(self, event):
        raise NotImplementedError

def load_plugins(package_name: str, ctx):
    mods = []
    for _, modname, ispkg in pkgutil.iter_modules(__import__(package_name, fromlist=['']).__path__):
        if ispkg:
            continue
        module = importlib.import_module(f"{package_name}.{modname}")
        for attr in dir(module):
            obj = getattr(module, attr)
            try:
                if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                    mods.append(obj(ctx))
            except Exception:
                pass
    return mods
