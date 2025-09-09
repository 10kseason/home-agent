import asyncio, os, sys, subprocess, platform, time, json, yaml
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from .schemas import Event, Result, PluginEventIn

# --- simple TTL-based dedup to avoid event loops ---
_DEDUP = {}

def _seen_recent(key: str, ttl: float = 3.0) -> bool:
    import time as _t
    now = _t.time()
    # purge occasionally
    if len(_DEDUP) > 2048:
        for k, until in list(_DEDUP.items())[:1024]:
            if until <= now:
                _DEDUP.pop(k, None)
    until = _DEDUP.get(key, 0.0)
    if until > now:
        return True
    _DEDUP[key] = now + ttl
    return False

CAPTURE_LOG_PATH = Path(__file__).resolve().parents[1] / "Capture-assist" / "capture_assist.log"

OVERLAY_HOST = os.environ.get("OVERLAY_HOST", "127.0.0.1")
OVERLAY_PORT = int(os.environ.get("OVERLAY_PORT", "8350"))
OVERLAY_BASE = f"http://{OVERLAY_HOST}:{OVERLAY_PORT}"
OVERLAY_EVENT_URL = os.environ.get("OVERLAY_EVENT_URL", f"{OVERLAY_BASE}/event")
OVERLAY_TOAST_URL = os.environ.get("OVERLAY_TOAST_URL", f"{OVERLAY_BASE}/overlay/event")


def _clear_capture_log(path: Path = CAPTURE_LOG_PATH) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _prune_capture_log(path: Path = CAPTURE_LOG_PATH, older_than: float = 60.0) -> None:
    now = time.time()
    try:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                ts, _rest = line.split("\t", 1)
                if float(ts) >= now - older_than:
                    kept.append(line)
            except Exception:
                continue
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    except Exception:
        pass


async def _capture_log_worker():
    while True:
        await asyncio.sleep(60)
        _prune_capture_log()

def _spawn_overlay(cfg):
    """Launch the Overlay using pathlib-based resolution.

    If absolute paths are not provided, fall back to repository-relative
    defaults so the config can remain portable.
    """
    ov = (cfg or {}).get("overlay", {}) or {}
    if not ov.get("enable"):
        logger.info("[overlay] disabled")
        return None

    repo_root = Path(__file__).resolve().parents[1]
    # Prefer overlay.python, fallback to top-level cfg['python'], then sys.executable
    py = ov.get("python") or (cfg.get("python") if isinstance(cfg, dict) else None) or sys.executable
    script = ov.get("script")
    if script:
        sp = Path(script)
        if not sp.is_absolute():
            sp = repo_root / sp
    else:
        sp = repo_root / "Overlay" / "overlay_app.py"
    if not sp.exists():
        # try cache
        try:
            cache = _load_paths_cache()
            csp = cache.get("overlay_script")
            if csp:
                csp = Path(csp)
                if csp.exists():
                    sp = csp
        except Exception:
            pass
    if not sp.exists():
        logger.warning(f"[overlay] script not found at {sp}; skip auto-launch")
        return None

    cwd = ov.get("cwd")
    cw = Path(cwd) if cwd else sp.parent
    if not cw.is_absolute():
        cw = (repo_root / cw).resolve()
    args = ov.get("args", []) or []

    creationflags = 0
    if platform.system() == "Windows" and ov.get("no_console", False):
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    cmd = [py, str(sp), *args]
    try:
        proc = subprocess.Popen(cmd, cwd=str(cw), creationflags=creationflags)
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

    repo_root = Path(__file__).resolve().parents[1]
    # Resolve working directory
    cwd_raw = spec.get("cwd")
    if cwd_raw:
        cw = Path(cwd_raw)
        if not cw.is_absolute():
            cw = (repo_root / cw).resolve()
    else:
        cw = repo_root

    # Resolve interpreter/command
    cmd_raw = spec.get("command")
    py = cmd_raw or (getattr(cfg, "get", lambda *_: None)("python")) or sys.executable
    try:
        if cmd_raw and not Path(cmd_raw).exists():
            py = sys.executable
    except Exception:
        py = sys.executable

    # Resolve script args
    args = list(spec.get("args", []) or [])
    if args:
        try:
            first = args[0]
            if isinstance(first, str) and first.lower().endswith(".py"):
                p = Path(first)
                if not p.is_absolute():
                    p = cw / p
                elif not p.exists():
                    # absolute but missing → try repo_root fallback
                    p = (repo_root / first.lstrip("/\\")).resolve()
                if not p.exists():
                    # try cache mapping
                    cache = _load_paths_cache()
                    key_map = {
                        "stt.start": "stt_script",
                        "ocr.start": "ocr_script",
                        "mictrans.start": "mictrans_script",
                        "capture_assist.start": "capture_assist_script",
                    }
                    ck = key_map.get(key)
                    if ck:
                        csp = cache.get(ck)
                        if csp and Path(csp).exists():
                            p = Path(csp)
                args[0] = str(p)
        except Exception:
            pass

    creationflags = 0
    if platform.system() == "Windows" and spec.get("no_console", False):
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    env = os.environ.copy()
    env.update(spec.get("env", {}) or {})

    cmd = [py, *args]
    try:
        proc = subprocess.Popen(cmd, cwd=str(cw), env=env, creationflags=creationflags)
        logger.info(f"[tool:{key}] launched pid={proc.pid} cmd={cmd}")
        return proc
    except Exception as e:
        logger.error(f"[tool:{key}] launch failed: {e}")
        return None

def create_app(ctx, plugins=None):
    def _venv_python() -> str | None:
        root = Path(__file__).resolve().parents[1]
        if platform.system() == "Windows":
            p = root / ".venv" / "Scripts" / "python.exe"
        else:
            p = root / ".venv" / "bin" / "python"
        return str(p) if p.exists() else None

    def _normalize_config_python() -> None:
        """Ensure a single python interpreter path is set and referenced.

        - Sets top-level cfg['python'] to the venv python if available.
        - Ensures overlay.python and tools.*.command reference this value
          when missing or pointing to a stale/missing interpreter.
        - Writes back to config.yaml if changes were made.
        """
        cfg = getattr(ctx, "config", {}) or {}
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "config.yaml"
        changed = False

        py_venv = _venv_python()
        py_cfg = cfg.get("python")

        # Set default python if not present and venv exists
        if not py_cfg and py_venv:
            cfg["python"] = py_venv
            py_cfg = py_venv
            changed = True

        # Overlay python fallback
        ov = cfg.get("overlay") or {}
        if py_cfg and not ov.get("python"):
            ov["python"] = py_cfg
            cfg["overlay"] = ov
            changed = True

        # Helper to decide if a command looks stale/missing
        def _needs_update(cmd: str | None) -> bool:
            if not cmd:
                return True
            try:
                return not Path(str(cmd)).exists()
            except Exception:
                return True

        # Update core tools to use a single interpreter when appropriate
        tools = cfg.get("tools") or {}
        core_keys = [
            "stt.start",
            "ocr.start",
            "mictrans.start",
            "capture_assist.start",
            "web.search",
        ]
        for k in core_keys:
            spec = tools.get(k) or {}
            if py_cfg and _needs_update(spec.get("command")):
                spec["command"] = py_cfg
                tools[k] = spec
                changed = True
        cfg["tools"] = tools

        if changed:
            try:
                # Persist and reflect in context
                cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                ctx.config = cfg
                logger.info("[config] normalized python interpreter paths and updated config.yaml")
            except Exception as e:
                logger.warning(f"[config] failed to write normalized config: {e}")
    def _discover_paths() -> dict:
        """Discover key tool directories relative to the repository root and return a mapping.

        The result is intended to be persisted on shutdown so future runs can reuse
        the discovered locations regardless of the working directory.
        """
        root = Path(__file__).resolve().parents[1]
        cand = {
            "stt": ["STT"],
            "ocr": ["OCR"],
            "mictrans": ["Mic-trans-assist", "Mic_trans_assist", "mictrans", "MicTrans"],
            "capture_assist": ["Capture-assist", "Capture_Assist", "capture_assist"],
            "overlay": ["Overlay"],
        }
        out = {"root": str(root)}
        for key, names in cand.items():
            found = None
            for name in names:
                p = root / name
                if p.exists() and p.is_dir():
                    found = p
                    break
            if not found:
                # fallback: shallow search (max depth 2)
                try:
                    for base, dirs, _ in os.walk(root):
                        depth = Path(base).relative_to(root).parts
                        if len(depth) > 1:
                            continue
                        for d in dirs:
                            if d.lower().replace("-", "_") == names[0].lower().replace("-", "_"):
                                found = Path(base) / d
                                break
                        if found:
                            break
                except Exception:
                    pass
            if found:
                out[key] = str(found)
                # attempt to detect primary script within the folder
                try:
                    scripts_map = {
                        "stt": ["VSRG-Ts-to-kr.py", "main.py"],
                        "ocr": ["main.py"],
                        "mictrans": ["mictrans.py"],
                        "capture_assist": ["capture_assist.py"],
                        "overlay": ["overlay_app.py"],
                    }
                    for cand_name in scripts_map.get(key, []):
                        sp = found / cand_name
                        if sp.exists():
                            out[f"{key}_script"] = str(sp)
                            break
                except Exception:
                    pass
        out["timestamp"] = time.time()
        return out


    # --- user path override + normalization helpers ---
    def _load_user_paths() -> dict:
        """Optional user-specified path map. Highest priority if present."""
        try:
            root = Path(__file__).resolve().parents[1]
            candidates = []
            # 1) explicit via env
            envp = os.environ.get("HOME_AGENT_PATHS")
            if envp:
                candidates.append(Path(envp))
            # 2) repo-local defaults
            candidates.append(root / "paths.user.json")
            candidates.append(root / "config" / "paths.json")
            for cp in candidates:
                try:
                    if cp and cp.exists():
                        data = json.loads(cp.read_text(encoding="utf-8"))
                        data["_source"] = str(cp)
                        return data
                except Exception:
                    # ignore bad JSON candidates
                    pass
        except Exception:
            pass
        return {}

    def _normalize_paths(data: dict) -> dict:
        """Make path-like values absolute based on 'root'; keep scalars as-is."""
        if not data:
            return {}
        try:
            base = data.get("root")
            if base:
                base = Path(base)
                if not base.is_absolute():
                    base = (Path(__file__).resolve().parents[1] / base).resolve()
            else:
                base = Path(__file__).resolve().parents[1]
            out = {"root": str(base.resolve())}
            for k, v in list(data.items()):
                if k in ("root",) or k.startswith("_"):
                    continue
                try:
                    pv = Path(v)
                    if not pv.is_absolute():
                        pv = base / pv
                    out[k] = str(pv.resolve())
                except Exception:
                    out[k] = v
            # carry source/metadata if any
            for k, v in data.items():
                if k.startswith("_"):
                    out[k] = v
            return out
        except Exception:
            return data

    def _merge_left_biased(*layers: dict) -> dict:
        """Merge dicts giving priority to left-most layer keys."""
        merged = {}
        for layer in layers[::-1]:  # right-most first, left-most last update wins
            if layer:
                merged.update(layer)
        return merged
    def _load_paths_cache() -> dict:
        root = Path(__file__).resolve().parents[1]
        user = _load_user_paths()
        cache_file = root / "paths.cache.json"
        cache = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        # baseline discovery
        disc = _discover_paths()

        # Normalize and merge with priority: user > cache > discovery
        merged = _merge_left_biased(
            _normalize_paths(user),
            _normalize_paths(cache),
            _normalize_paths(disc),
        )
        return merged

    

    async def _restart_if_cache_missing():
        """If paths.cache.json is missing, create it and restart the server process.
        We now respect a user-provided paths file. If present, we skip restart.
        """
        try:
            # If user file exists, skip restart entirely
            if _load_user_paths():
                logger.info("[paths] user paths present; skip restart")
                return
            root = Path(__file__).resolve().parents[1]
            cache = root / "paths.cache.json"
            if not cache.exists():
                data = _discover_paths()
                cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"[paths] created path cache at {cache}; restarting agent")
                await asyncio.sleep(0.2)
                os.execv(sys.executable, [sys.executable, "-m", "agent.main"])
        except Exception as e:
            logger.debug(f"[paths] restart-if-missing error: {e}")
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        # Ensure path cache exists before wiring anything else
        await _restart_if_cache_missing()
        # Ensure a single python interpreter setting is present and referenced
        _normalize_config_python()
        app.state.overlay_proc = _spawn_overlay(getattr(ctx, "config", {}))
        app.state.stt_proc = None
        app.state.mictrans_proc = None
        app.state.ocr_proc = None
        app.state.capture_proc = None
        app.state.assist_mode = bool(getattr(ctx, "assist_mode", False))
        app.state.assist_task = None
        app.state.capture_log_task = asyncio.create_task(_capture_log_worker())
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

            # If the enhanced overlay sink plugin isn't present, install a minimal
            # fallback forwarder so STT/OCR/MicTrans/Capture-Assist results appear
            # on the Overlay. Keeps behavior when plugin is available.
            has_overlay_sink = any(getattr(p, "name", "") == "enhanced_overlay_sink" for p in (plugins or []))
            force_fallback = bool(os.environ.get("OVERLAY_FORCE_FALLBACK"))

            # Runtime policy: block OCR while high‑VRAM STT (VSRG translator) is running.
            # Env override: BLOCK_OCR_WHILE_STT=0 disables; any other/non-empty enables.
            def _flag_true(v):
                s = str(v).strip().lower()
                return s not in ("0", "false", "no", "off", "")
            cfg_orch = (getattr(ctx, "config", {}).get("orchestrator") or {})
            cfg_block = cfg_orch.get("block_ocr_while_stt", True)
            env_block = os.environ.get("BLOCK_OCR_WHILE_STT")
            block_ocr_while_stt = _flag_true(env_block) if env_block is not None else bool(cfg_block)

            async def _forward_to_overlay(event_type: str, payload: dict, priority: int = 5):
                try:
                    # Never re-forward already normalized result types to avoid self-triggering
                    if event_type.endswith(".result"):
                        return
                    # Build normalized event for the Overlay app
                    evt: dict
                    if event_type.startswith("stt.") or event_type.startswith("mictrans."):
                        text = (payload or {}).get("text", "")
                        translation = (payload or {}).get("translation", "")
                        confidence = (payload or {}).get("confidence", 0)
                        display = (translation or text or "").strip()
                        evt = {
                            "type": "stt.result",
                            "payload": {
                                "text": display[:300],
                                "original": text,
                                "translation": translation,
                                "confidence": confidence,
                                "assist": bool((payload or {}).get("assist", False)),
                            },
                            "priority": priority,
                            "timestamp": int(time.time() * 1000),
                            "source": "agent",
                        }
                    elif event_type.startswith("ocr.") or event_type.startswith("capture_assist."):
                        text = (payload or {}).get("text") or (payload or {}).get("ocr") or ""
                        bbox = (payload or {}).get("bbox", [])
                        confidence = (payload or {}).get("confidence", 0)
                        evt_type = "capture_assist.result" if event_type.startswith("capture_assist.") else "ocr.result"
                        evt = {
                            "type": evt_type,
                            "payload": {
                                "text": (text or "")[:300],
                                "bbox": bbox,
                                "confidence": confidence,
                                "assist": bool((payload or {}).get("assist", False)),
                            },
                            "priority": priority,
                            "timestamp": int(time.time() * 1000),
                            "source": "agent",
                        }
                    else:
                        return

                    # Post to Overlay /event (async via thread to avoid blocking)
                    async def _post():
                        try:
                            import requests as _rq  # prefer widely-available requests
                            _rq.post(OVERLAY_EVENT_URL, json=evt, timeout=3)
                        except Exception as _e:
                            logger.debug(f"[overlay-forward] post failed: {_e}")
                    await _post()
                except Exception as e:
                    logger.debug(f"[overlay-forward] error: {e}")

            if force_fallback or not has_overlay_sink:
                async def _overlay_fallback_handler(ev):
                    # Drop events that are already post-processed results to avoid echo loops
                    if ev.type.endswith(".result") or getattr(ev, "source", "") == "overlay":
                        return
                    payload = getattr(ev, "payload", {}) or {}
                    # Dedup by type+text-ish payload within a short TTL window
                    text = (payload.get("translation") or payload.get("text") or payload.get("ocr") or "")
                    key = f"ovf:{ev.type}:{text[:200]}"
                    if _seen_recent(key, ttl=5.0):
                        return
                    await _forward_to_overlay(ev.type, payload, getattr(ev, "priority", 5))
                # Always forward mictrans.* (lightweight path) as a safety net.
                # Avoid duplicate capture_assist forwarding when sink is present.
                prefixes = ["mictrans.", "stt.", "ocr."]
                if not has_overlay_sink:
                    prefixes.append("capture_assist.")
                for _prefix in prefixes:
                    ctx.bus.subscribe(_prefix, _overlay_fallback_handler)
                    app.state._plugin_unsubs.append((_prefix, _overlay_fallback_handler))
                logger.info(
                    "[overlay-forward] Installed minimal overlay forwarder (%s)" % (
                        "forced" if force_fallback else "plugin not found"
                    )
                )

            # MicTrans feed bridge: Ensure mictrans.text shows up on overlay feed
            # even if the sink is slow or missing. Duplicates are dropped by
            # the overlay app's duplicate filter (5s window by text hash).
            async def _mictrans_bridge(ev):
                if ev.type != "mictrans.text":
                    return
                payload = getattr(ev, "payload", {}) or {}
                key = f"micbridge:{payload.get('text','')[:200]}:{payload.get('translation','')[:200]}"
                if _seen_recent(key, ttl=5.0):
                    return
                await _forward_to_overlay(ev.type, payload, getattr(ev, "priority", 5))

            ctx.bus.subscribe("mictrans.text", _mictrans_bridge)
            app.state._plugin_unsubs.append(("mictrans.text", _mictrans_bridge))

            async def _toast_handler(ev):
                payload = ev.payload or {}
                # avoid infinite loops if we re-publish the toast event
                if payload.get("_relay"):
                    return
                # Drop obvious duplicates within a short window
                title = payload.get("title", "")
                text = payload.get("text", "")
                if _seen_recent(f"toast:{title}:{text}", ttl=5.0):
                    return
                logger.info(f"[toast] {title}: {text}")
                # relay event on the bus so overlay sinks can forward the message
                try:
                    await ctx.bus.publish(
                        Event(
                            type="overlay.toast",
                            payload={**payload, "_relay": True},
                            priority=getattr(ev, "priority", 5),
                        )
                    )
                except Exception as e:
                    logger.debug(f"[toast] relay failed: {e}")
                # Fallback: when enhanced overlay sink is missing, post directly
                if force_fallback or not has_overlay_sink:
                    try:
                        import requests as _rq
                        _rq.post(OVERLAY_TOAST_URL, json={
                            "type": "overlay.toast",
                            "payload": {**payload, "_relay": True},
                            "priority": getattr(ev, "priority", 5),
                            "timestamp": int(time.time() * 1000),
                            "source": "agent",
                        }, timeout=3)
                    except Exception as e:
                        logger.debug(f"[toast] http fallback failed: {e}")

            ctx.bus.subscribe("overlay.toast", _toast_handler)
            app.state._plugin_unsubs.append(("overlay.toast", _toast_handler))

            async def _stt_handler(ev):
                if ev.type == "stt.start":
                    if app.state.stt_proc and app.state.stt_proc.poll() is None:
                        logger.info("[stt] already running")
                    else:
                        app.state.stt_proc = _spawn_tool(getattr(ctx, "config", {}), "stt.start")
                        logger.info("[stt] started basic STT (VSRG-Ts-to-kr.py)")
                        # If policy enabled, stop heavy OCR pipelines to save VRAM
                        if block_ocr_while_stt:
                            try:
                                await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                                app.state.ocr_proc = None
                            except Exception:
                                pass
                            try:
                                await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
                                app.state.capture_proc = None
                            except Exception:
                                pass
                            # Inform user via toast (relayed to overlay)
                            try:
                                await ctx.bus.publish(Event(
                                    type="overlay.toast",
                                    payload={"title": "VRAM 보호", "text": "STT 실행 중이라 OCR이 중지됩니다."},
                                    priority=5,
                                ))
                            except Exception:
                                pass

                elif ev.type == "stt.stop":
                    await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
                    app.state.stt_proc = None
                    logger.info("[stt] stopped")

                elif ev.type == "mictrans.start":
                    if app.state.mictrans_proc and app.state.mictrans_proc.poll() is None:
                        await _terminate_proc(app.state.mictrans_proc, name="mictrans", timeout=3.0)
                    app.state.mictrans_proc = _spawn_tool(getattr(ctx, "config", {}), "mictrans.start")
                    logger.info("[mictrans] started voice input (Whisper)")

                elif ev.type == "mictrans.stop":
                    await _terminate_proc(getattr(app.state, "mictrans_proc", None), name="mictrans", timeout=3.0)
                    app.state.mictrans_proc = None
                    logger.info("[mictrans] stopped")

            ctx.bus.subscribe("stt.", _stt_handler)
            ctx.bus.subscribe("mictrans.", _stt_handler)
            app.state._plugin_unsubs.append(("stt.", _stt_handler))
            app.state._plugin_unsubs.append(("mictrans.", _stt_handler))

            async def _ocr_handler(ev):
                # Separate pipelines: 'ocr.*' (VLM OCR + translate) vs 'capture_assist.*' (EasyOCR only)
                if ev.type == "ocr.start":
                    if block_ocr_while_stt and app.state.stt_proc and app.state.stt_proc.poll() is None:
                        logger.info("[ocr] blocked: STT running (VRAM policy)")
                        try:
                            await ctx.bus.publish(Event(
                                type="overlay.toast",
                                payload={"title": "OCR 차단", "text": "STT 실행 중이라 OCR을 시작하지 않습니다."},
                                priority=5,
                            ))
                        except Exception:
                            pass
                    elif app.state.ocr_proc and app.state.ocr_proc.poll() is None:
                        logger.info("[ocr] already running")
                    else:
                        app.state.ocr_proc = _spawn_tool(getattr(ctx, "config", {}), "ocr.start")
                        logger.info("[ocr] started VLM OCR pipeline")
                elif ev.type == "ocr.stop":
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
                    app.state.ocr_proc = None
                    app.state.capture_proc = None
                    logger.info("[ocr] stopped")
                elif ev.type == "capture_assist.start":
                    if block_ocr_while_stt and app.state.stt_proc and app.state.stt_proc.poll() is None:
                        logger.info("[capture_assist] blocked: STT running (VRAM policy)")
                        try:
                            await ctx.bus.publish(Event(
                                type="overlay.toast",
                                payload={"title": "캡처 차단", "text": "STT 실행 중이라 캡처를 시작하지 않습니다."},
                                priority=5,
                            ))
                        except Exception:
                            pass
                    elif app.state.capture_proc and app.state.capture_proc.poll() is None:
                        logger.info("[capture_assist] already running")
                    else:
                        app.state.capture_proc = _spawn_tool(getattr(ctx, "config", {}), "capture_assist.start")
                        logger.info("[capture_assist] started")
                elif ev.type == "capture_assist.stop":
                    await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
                    app.state.capture_proc = None
                    logger.info("[capture_assist] stopped")

            ctx.bus.subscribe("ocr.", _ocr_handler)
            ctx.bus.subscribe("capture_assist.", _ocr_handler)
            app.state._plugin_unsubs.append(("ocr.", _ocr_handler))
            app.state._plugin_unsubs.append(("capture_assist.", _ocr_handler))

            async def _assist_handler(ev):
                if ev.type == "assist.on":
                    app.state.assist_mode = True
                    ctx.assist_mode = True
                    # 기존 프로세스 정리만 하고 자동 시작하지 않음
                    await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "mictrans_proc", None), name="mictrans", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
                    app.state.stt_proc = None
                    app.state.mictrans_proc = None
                    app.state.ocr_proc = None
                    app.state.capture_proc = None

                    # 보조모드 활성화됨 - 수동으로 도구 시작 필요
                    logger.info("[assist] 보조모드 활성화 - 명시적 명령으로 도구 시작 필요")
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
                    await _terminate_proc(getattr(app.state, "mictrans_proc", None), name="mictrans", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                    await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
                    app.state.stt_proc = None
                    app.state.mictrans_proc = None
                    app.state.ocr_proc = None
                    app.state.capture_proc = None
                    logger.info("[assist] 보조모드 종료 - 수동으로 도구 시작 필요")

            ctx.bus.subscribe("assist.", _assist_handler)
            app.state._plugin_unsubs.append(("assist.", _assist_handler))

            async def _cmd_handler(ev):
                # ``cmd.detected``는 payload 에서 명령을 추출해야 하므로 먼저 검사
                if ev.type == "cmd.detected":
                    c = (ev.payload or {}).get("cmd")
                # 그 외 ``cmd.*`` 타입은 이벤트명으로부터 명령을 파싱
                elif ev.type.startswith("cmd."):
                    c = ev.type.split(".", 1)[1]
                else:
                    c = None

                if not c:
                    return

                # 중복 억제용 타임스탬프
                import time as _t
                ts = ev.payload.get("ts") if isinstance(ev.payload, dict) else None
                if not ts:
                    ts = _t.time()

                # 라우팅
                if c == "capture":
                    await ctx.bus.publish(Event(
                        type="capture_assist.start",
                        payload={"reason": "voice_cmd", "cmd": c, "ts": ts},
                        priority=2,
                        source="router",
                        timestamp=_t.time(),
                    ))
                elif c == "assist":
                    await ctx.bus.publish(Event(
                        type="assist.on",
                        payload={"ts": ts},
                        priority=5,
                        source="router",
                        timestamp=_t.time(),
                    ))
                elif c == "stop":
                    # 보조모드 종료 + OCR/STM 정리
                    await ctx.bus.publish(Event(
                        type="assist.off",
                        payload={"ts": ts},
                        priority=5,
                        source="router",
                        timestamp=_t.time(),
                    ))
                    await ctx.bus.publish(Event(type="capture_assist.stop", payload={}, priority=3, source="router", timestamp=_t.time()))
                    await ctx.bus.publish(Event(type="mictrans.stop", payload={}, priority=3, source="router", timestamp=_t.time()))
                # 필요하면 여기서 summarize/translate 등도 매핑

            # 명령 라우팅 활성화 여부
            cmd_cfg = (getattr(ctx, "config", {}).get("commands") or {})
            enable_detected = cmd_cfg.get("enable_detected")
            enable_detected = True if enable_detected is None else bool(enable_detected)
            router_handle_detected = bool(cmd_cfg.get("router_handle_detected", False))
            if enable_detected:
                # Always handle explicit cmd.* topics (e.g., cmd.capture)
                ctx.bus.subscribe("cmd.", _cmd_handler)
                app.state._plugin_unsubs.append(("cmd.", _cmd_handler))
                # Optionally also handle cmd.detected (default off to avoid
                # duplication with AssistCommandPlugin)
                if router_handle_detected:
                    ctx.bus.subscribe("cmd.detected", _cmd_handler)
                    app.state._plugin_unsubs.append(("cmd.detected", _cmd_handler))
                    logger.info("[router] subscribed to 'cmd.*' and 'cmd.detected'")
                else:
                    logger.info("[router] subscribed to 'cmd.*' (plugin handles cmd.detected)")
            else:
                logger.info("[router] command routing disabled by config")

            # Tool gateway handler for LLM responses
            async def _llm_handler(ev):
                if ev.type == "llm.response":
                    logger.info(f"[server] Received llm.response event: {ev.payload}")
                    from .tool_gateway import process_response
                    await process_response(ev.payload or {}, ctx, ctx.bus)
                    
            ctx.bus.subscribe("llm.response", _llm_handler)
            app.state._plugin_unsubs.append(("llm.response", _llm_handler))
            logger.info("[tool-gw] subscribed to 'llm.response'")

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

            # Shutdown STT and MicTrans
            try:
                await _terminate_proc(getattr(app.state, "stt_proc", None), name="stt", timeout=3.0)
                await _terminate_proc(getattr(app.state, "mictrans_proc", None), name="mictrans", timeout=3.0)
            except Exception as e:
                logger.debug(f"[lifespan] stt terminate error: {e}")
            app.state.stt_proc = None
            app.state.mictrans_proc = None

            # Shutdown OCR
            try:
                await _terminate_proc(getattr(app.state, "ocr_proc", None), name="ocr", timeout=3.0)
                await _terminate_proc(getattr(app.state, "capture_proc", None), name="capture_assist", timeout=3.0)
            except Exception as e:
                logger.debug(f"[lifespan] ocr terminate error: {e}")
            app.state.ocr_proc = None
            app.state.capture_proc = None

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

            # Stop capture log cleaner
            try:
                task = getattr(app.state, "capture_log_task", None)
                if task:
                    task.cancel()
                    try:
                        await task
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[lifespan] capture log task cancel error: {e}")
            app.state.capture_log_task = None

            _clear_capture_log()

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

    # Persist discovered tool paths on shutdown
    async def _persist_discovered_paths():
        try:
            data = _discover_paths()
            root = Path(data.get("root", Path(__file__).resolve().parents[1]))
            path = root / "paths.cache.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[paths] persisted discovered paths to {path}")
        except Exception as e:
            logger.debug(f"[paths] persist error: {e}")

    # Register atexit-like callback via FastAPI shutdown event
    @app.on_event("shutdown")
    async def _on_shutdown():
        await _persist_discovered_paths()

    @app.get("/overlay/status")
    async def overlay_status():
        proc = getattr(app.state, "overlay_proc", None)
        running = bool(proc and proc.poll() is None)
        return {"ok": True, "running": running, "pid": (proc.pid if running else None)}

    @app.get("/assist/status")
    async def assist_status():
        return {"ok": True, "assist_mode": bool(getattr(app.state, "assist_mode", False))}

    return app
