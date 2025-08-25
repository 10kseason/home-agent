import asyncio
from loguru import logger
import uvicorn
from .context import load_config, Ctx
from .server import create_app
from .plugins import load_plugins
from .server import create_app
async def bus_loop(ctx: Ctx, plugins):
    # 구독 등록
    for p in plugins:
        for prefix in p.handles:
            ctx.bus.subscribe(prefix, p.handle)
            logger.info(f"Subscribed: {p.name} to {prefix}")
    await ctx.bus.run()

def main():
    config = load_config()
    ctx = Ctx(config)
    plugins = load_plugins("agent.plugins", ctx)
    app = create_app(ctx, plugins)
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])

if __name__ == "__main__":
    main()
