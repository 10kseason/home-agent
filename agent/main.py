import uvicorn
from loguru import logger
from .context import load_config, Ctx
from .server import create_app

def _load_plugins(cfg, ctx):
    try:
        # Import the package that contains plugin modules
        from . import plugins as plugins_pkg  # package
        from .plugins import load_plugins     # loader in __init__.py
    except Exception as e:
        logger.warning(f"[plugins] loader/package not found: {e}")
        return []

    try:
        package_name = plugins_pkg.__name__   # e.g., "agent.plugins"
        mods = load_plugins(package_name, ctx)
        logger.info(f"[plugins] loaded {len(mods)} plugin(s) from '{package_name}'")
        return mods
    except Exception as e:
        logger.error(f"[plugins] load error: {e}")
        return []

def main():
    cfg = load_config()
    ctx = Ctx(cfg)
    plugins = _load_plugins(cfg, ctx)  # list[BasePlugin]
    app = create_app(ctx, plugins)

    host = (cfg.get("server") or {}).get("host", "127.0.0.1")
    port = int((cfg.get("server") or {}).get("port", 8765))

    logger.info(f"[server] starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
