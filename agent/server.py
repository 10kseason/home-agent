import asyncio, os, sys, subprocess, platform, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from .schemas import Event, Result, PluginEventIn

def _spawn_overlay(cfg):
    ov = (cfg or {}).get("overlay", {}) or {}
    if not ov.get("enable"):
        logger.info("[overlay] disabled")
        return None

    py = ov.get("python") or sys.executable
    script = ov.get("script")
    if not script or not os.path.exists(script):
        logger.warning("[overlay] script not set or not found; skip auto-launch")
        return None

    cwd = ov.get("cwd") or os.path.dirname(script)
    args = ov.get("args", []) or []

    creationflags = 0
    if platform.system() == "Windows" and ov.get("no_console", False):
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    cmd = [py, script, *args]
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, creationflags=creationflags)
        logger.info(f"[overlay] launched pid={proc.pid} cmd={cmd}")
        return proc
    except Exception as e:
        logger.error(f"[overlay] launch failed: {e}")
        return None

async def _terminate_proc(proc, name="overlay", timeout: float = 3.0):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
        logger.info(f"[{name}] terminated")
    except Exception as e:
        logger.debug(f"[{name}] terminate error: {e}")


def _spawn_tool(cfg, key: str):
    tools = (cfg or {}).get("tools", {}) or {}
    spec = tools.get(key, {}) or {}
    if spec.get("kind") != "process":
        logger.info(f"[tool:{key}] disabled")
        return None

    py = spec.get("command") or sys.executable
    args = spec.get("args", []) or []
    cwd = spec.get("cwd") or os.getcwd()

    creationflags = 0
    if platform.system() == "Windows" and spec.get("no_console", False):
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    env = os.environ.copy()
    env.update(spec.get("env", {}) or {})

    cmd = [py, *args]
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=creationflags)
        logger.info(f"[tool:{key}] launched pid={proc.pid} cmd={cmd}")
        return proc
    except Exception as e:
        logger.error(f"[tool:{key}] launch failed: {e}")
        return None

def create_app(ctx, plugins=None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        app.state.overlay_proc = _spawn_overlay(getattr(ctx, "config", {}))
        app.state.stt_proc = None
        app.state.ocr_proc = None
        app.state.assist_mode = bool(getattr(ctx, "assist_mode", False))
        app.state.assist_task = None
        # Watch overlay process so the agent exits when overlay closes
        if app.state.overlay_proc:
            async def _overlay_watch():
                proc = app.state.overlay_proc
                try:
                    await asyncio.to_thread(proc.wait)
                    logger.info("[overlay] exited; shutting down agent server")
                    import os, signal
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception as e:
                    logger.debug(f"[overlay] watch error: {e}")
            app.state._overlay_watch = asyncio.create_task(_overlay_watch())
        else:
            app.state._overlay_watch = None
        try:
            # Wire plugins into the event bus
            app.state._plugin_unsubs = []
            for p in (plugins or []):
                for prefix in getattr(p, "handles", []) or []:
                    async def _handler(ev, _p=p):
                        try:
                            await _p.handle(ev)
                        except Exception as e:
                            logger.error(f"[plugin:{getattr(_p, 'name', _p)}] handle error: {e}")
                    ctx.bus.subscribe(prefix, _handler)
                    app.state._plugin_unsubs.append((prefix, _handler))
                    logger.info(f"[plugin] subscribed '{getattr(p,'name',p)}' to '{prefix}'")

            async def _stt_handler(ev):
                if ev.type == "stt.start":
                    if app.state.stt_proc and app.state.stt_proc.poll() is None:
                        logger.info("[stt] already running")
                    else:
                        key = "stt_assist.start" if app.state.assist_mode else "stt.start"
                        app.state.stt_proc = _spawn_tool(getattr(ctx, "config", {}), key)
                elif ev.type == "stt.stop":
                    if app.state.assist_mode:
                        logger.info("[stt] stop ignored in assist mode")
                    else:
                        await _terminate_proc(
                            getattr(app.state, "stt_proc", None),
                            name="stt",
                            timeout=3.0,
                        )
                        app.state.stt_proc = None

            ctx.bus.subscribe("stt.", _stt_handler)
            app.state._plugin_unsubs.append(("stt.", _stt_handler))

            async def _ocr_handler(ev):
                if ev.type == "ocr.start":
                    if app.state.ocr_proc and app.state.ocr_proc.poll() is None:
                        logger.info("[ocr] already running")
                    else:
                        key = "ocr_assist.start" if app.state.assist_mode else "ocr.start"
                        app.state.ocr_proc = _spawn_tool(getattr(ctx, "config", {}), key)
                elif ev.type == "ocr.stop":
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    app.state.ocr_proc = None

            ctx.bus.subscribe("ocr.", _ocr_handler)
            app.state._plugin_unsubs.append(("ocr.", _ocr_handler))

            async def _assist_handler(ev):
                if ev.type == "assist.on":
                    app.state.assist_mode = True
                    ctx.assist_mode = True
                    await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    app.state.stt_proc = _spawn_tool(getattr(ctx, "config", {}), "stt_assist.start")
                    app.state.ocr_proc = _spawn_tool(getattr(ctx, "config", {}), "ocr_assist.start")

                    if app.state.assist_task is None:
                        async def _ticker():
                            while app.state.assist_mode:
                                proc = getattr(app.state, "stt_proc", None)
                                if proc is None or proc.poll() is not None:
                                    logger.info("[assist] restarting stt_assist")
                                    app.state.stt_proc = _spawn_tool(
                                        getattr(ctx, "config", {}),
                                        "stt_assist.start",
                                    )
                                await asyncio.sleep(5)
                        app.state.assist_task = asyncio.create_task(_ticker())
                elif ev.type == "assist.off":
                    app.state.assist_mode = False
                    ctx.assist_mode = False
                    task = getattr(app.state, "assist_task", None)
                    if task:
                        task.cancel()
                        try:
                            await task
                        except Exception:
                            pass
                    app.state.assist_task = None
                    await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    app.state.stt_proc = _spawn_tool(getattr(ctx, "config", {}), "stt.start")
                    app.state.ocr_proc = None

            ctx.bus.subscribe("assist.", _assist_handler)
            app.state._plugin_unsubs.append(("assist.", _assist_handler))

            # Start event bus loop
            app.state.bus_task = asyncio.create_task(ctx.bus.run())
            yield
        finally:
            # Unsubscribe plugins
            try:
                for prefix, h in getattr(app.state, "_plugin_unsubs", []):
                    try:
                        ctx.bus.unsubscribe(prefix, h)
                    except Exception:
                        pass
                app.state._plugin_unsubs = []
            except Exception:
                pass
            # Stop overlay watcher
            try:
                watch = getattr(app.state, "_overlay_watch", None)
                if watch:
                    watch.cancel()
                    try:
                        await watch
                    except Exception:
                        pass
                app.state._overlay_watch = None
            except Exception:
                pass
            # Stop event bus loop
            try:
                bus_task = getattr(app.state, "bus_task", None)
                if bus_task:
                    bus_task.cancel()
                    try:
                        await bus_task
                    except Exception:
                        pass
                app.state.bus_task = None
            except Exception as e:
                logger.debug(f"[lifespan] bus task cancel error: {e}")
            # Shutdown overlay
            try:
                await _terminate_proc(getattr(app.state, "overlay_proc", None), name="overlay", timeout=3.0)
            except Exception as e:
                logger.debug(f"[lifespan] overlay terminate error: {e}")
            app.state.overlay_proc = None

            # Shutdown STT
            try:
                await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
            except Exception as e:
                logger.debug(f"[lifespan] stt terminate error: {e}")
            app.state.stt_proc = None

            # Shutdown OCR
            try:
                await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
            except Exception as e:
                logger.debug(f"[lifespan] ocr terminate error: {e}")
            app.state.ocr_proc = None

            # Cancel assist ticker
            try:
                task = getattr(app.state, "assist_task", None)
                if task:
                    task.cancel()
                    try:
                        await task
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[lifespan] assist task cancel error: {e}")
            app.state.assist_task = None

    app = FastAPI(title="Luna Local Agent", lifespan=lifespan)

    # CORS for external plugins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simple header auth (optional)
    def _auth(x_key: str | None = Header(default=None, alias="X-Agent-Key")):
        want = (getattr(ctx, "config", {}).get("server") or {}).get("api_key")
        if want and x_key != want:
            raise HTTPException(status_code=401, detail="invalid api key")

    # ---- Health & root ----
    @app.get("/")
    async def root():
        return {"ok": True, "name": "Luna Local Agent"}

    @app.get("/health")
    async def health():
        proc = getattr(app.state, "overlay_proc", None)
        running = bool(proc and proc.poll() is None)
        return {"ok": True, "overlay_running": running}

    # Ingest fully-formed internal Event
    @app.post("/event")
    async def ingest_event(ev: Event):
        try:
            await ctx.bus.publish(ev)
            return JSONResponse(Result(ok=True, message="queued").model_dump())
        except Exception as ex:
            logger.exception("Failed to enqueue event")
            raise HTTPException(status_code=500, detail=str(ex))

    # Plugin ingress (loose schema → normalize to Event)
    plugin = APIRouter(prefix="/plugin")

    @plugin.post("/event")
    async def plugin_event(body: PluginEventIn, _=Depends(_auth)):
        ev = Event(
            type=body.type,
            payload=body.payload or {},
            priority=int(body.priority or 5),
            source=body.source or "plugin",
            timestamp=time.time(),
        )
        await ctx.bus.publish(ev)
        return {"ok": True, "message": "queued"}

    @plugin.post("/events")
    async def plugin_events(bodies: list[PluginEventIn], _=Depends(_auth)):
        for b in bodies:
            ev = Event(
                type=b.type,
                payload=b.payload or {},
                priority=int(b.priority or 5),
                source=b.source or "plugin",
                timestamp=time.time(),
            )
            await ctx.bus.publish(ev)
        return {"ok": True, "message": f"queued {len(bodies)} events"}

    app.include_router(plugin)

    # --- overlay control (optional) ---
    @app.post("/overlay/start")
    async def overlay_start():
        proc = getattr(app.state, "overlay_proc", None)
        if proc and proc.poll() is None:
            return {"ok": True, "status": "already_running", "pid": proc.pid}
        app.state.overlay_proc = _spawn_overlay(getattr(ctx, "config", {}))
        if app.state.overlay_proc:
            return {"ok": True, "status": "launched", "pid": app.state.overlay_proc.pid}
        raise HTTPException(status_code=500, detail="overlay launch failed")

    @app.post("/overlay/stop")
    async def overlay_stop():
        await _terminate_proc(getattr(app.state, "overlay_proc", None), name="overlay", timeout=3.0)
        app.state.overlay_proc = None
        return {"ok": True, "status": "stopped"}

    @app.get("/overlay/status")
    async def overlay_status():
        proc = getattr(app.state, "overlay_proc", None)
        running = bool(proc and proc.poll() is None)
        return {"ok": True, "running": running, "pid": (proc.pid if running else None)}

    @app.get("/assist/status")
    async def assist_status():
        return {"ok": True, "assist_mode": bool(getattr(app.state, "assist_mode", False))}

    return app
