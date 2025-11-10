import logging
import traceback
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
import os
import types

import numpy as np

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        _UNICODE_ENABLED = True
    except Exception:
        _UNICODE_ENABLED = False
else:
    _UNICODE_ENABLED = True

_BAR_CHAR_FILLED = "█" if _UNICODE_ENABLED else "#"
_BAR_CHAR_EMPTY = "░" if _UNICODE_ENABLED else "-"
_ARROW = "▶" if _UNICODE_ENABLED else ">"

def setup_logger(
    service_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> Logger:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.getLogger("kubernetes.client.rest").setLevel(logging.WARNING)
    logging.getLogger("kubernetes").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    def emit_utf8(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    console_handler.emit = types.MethodType(emit_utf8, console_handler)
    logger.addHandler(console_handler)

    if log_to_file:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir_time = Path(log_dir) / now
        log_dir_time.mkdir(parents=True, exist_ok=True)

        log_file = log_dir_time / f"{service_name}_{now}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10 * 1024 * 1024, 
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))

def _bar(pct: float, width: int = 12) -> str:
    pct = _clamp(pct)
    filled = round(pct / 100 * width)
    return _BAR_CHAR_FILLED * filled + _BAR_CHAR_EMPTY * (width - filled)

def _color(v: float, warn: float, crit: float, reverse: bool = False) -> str:
    GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"
    ok = v <= warn if reverse else v < warn
    mid = (warn < v <= crit) if reverse else (warn <= v < crit)
    return GREEN if ok else (YELLOW if mid else RED)

def _fmt_pct(v: float) -> str:
    try:
        return f"{float(v):6.2f}%"
    except Exception:
        return str(v)

def _fmt_ms(v: float) -> str:
    MS_TO_SECONDS_THRESHOLD = 1000.0
    try:
        v = float(v)
        if v < 1.0:
            return f"{v * 1000:6.2f}µs"
        if v < MS_TO_SECONDS_THRESHOLD:
            return f"{v:6.2f}ms"
        return f"{v / MS_TO_SECONDS_THRESHOLD:6.2f}s"
    except Exception:
        return str(v)

def _safe_q_values(
    agent: Any, state_key
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
    try:
        q_table = getattr(agent, "q_table", None)
        if q_table is None or len(q_table) == 0:
            return None, None, None
        
        agent_type = getattr(agent, "agent_type", "").upper()
        if agent_type not in ("Q", "QFUZZYHYBRID"):
            return None, None, None
        
        if isinstance(state_key, np.ndarray):
            state_key = tuple(state_key.flatten())
        
        if state_key not in q_table:
            return None, None, None
        
        q = q_table[state_key]
        max_q = float(np.max(q))
        best_idx = int(np.argmax(q))
        
        return q, max_q, best_idx
        
    except Exception as e:
        return None, None, None

def log_verbose_details(
    observation: Dict[str, Any], agent: Any, verbose: bool, logger: Logger
) -> None:
    if not verbose:
        return

    try:
        cpu = float(observation.get("cpu_usage", 0.0))
        mem = float(observation.get("memory_usage", 0.0))
        rt = float(observation.get("response_time", 0.0))
        act = observation.get("last_action", 0)
        iter_no = observation.get("iteration")

        cpu_col = _color(cpu, warn=70, crit=90)
        mem_col = _color(mem, warn=75, crit=90)
        rt_col = _color(rt, warn=70, crit=90)

        cpu_bar = _bar(cpu)
        mem_bar = _bar(mem)
        rt_bar = _bar(rt)

        state_key = agent.get_state_key(observation)
        q_vals, qmax, best_idx = _safe_q_values(agent, state_key)

        RESET = "\033[0m"
        hdr = f"{_ARROW} Iter {iter_no:02d} " if isinstance(iter_no, int) else f"{_ARROW} "
        cpu_str = f"{cpu_col}CPU {_fmt_pct(cpu)} {cpu_bar}{RESET}"
        mem_str = f"{mem_col}MEM {_fmt_pct(mem)} {mem_bar}{RESET}"
        rt_str = f"{rt_col}RT {_fmt_pct(rt)} {rt_bar}{RESET}"
        act_str = f"ACT {int(act):3d}"

        if qmax is not None and best_idx is not None:
            q_str = f"Qmax {qmax:+.3f}"
            best_s = f"Best {best_idx:3d}"
        else:
            q_str, best_s = "Qmax  n/a", "Best  n/a"

        logger.info(
            f"{hdr}| {cpu_str} | {mem_str} | {rt_str} | {act_str} | {q_str} | {best_s}"
        )
        
    except Exception as e:
        logger.warning(f"Error in verbose logging: {e}")
        logger.debug(f"Observation: {observation}")