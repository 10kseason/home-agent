import importlib, pkgutil
from typing import List, Type

class BasePlugin:
    name: str = "base"
    handles: List[str] = []  # event type prefixes this plugin can handle

    def __init__(self, ctx):
        self.ctx = ctx  # access to bus, policy, config, sinks, http client

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
