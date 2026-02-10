import logging
import os
import sys
import types
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

if os.name == "nt":
    try:
        import io

        if isinstance(sys.stdout, io.TextIOWrapper):
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
    console_handler.setLevel(numeric_level)
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
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
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
    """
    Displays system metrics, request rate, trends, and Q-learning information.
    """
    if not verbose:
        return

    try:
        cpu = float(observation.get("cpu_usage", 0.0))
        mem = float(observation.get("memory_usage", 0.0))
        rt = float(observation.get("response_time", 0.0))
        rt_ms = float(observation.get("response_time_ms", 0.0))
        act = observation.get("last_action", 0)
        iter_no = observation.get("iteration")

        req_rate_norm = float(observation.get("request_rate_normalized", 0.0))
        req_rate_raw = float(observation.get("request_rate", 0.0))
        req_trend = observation.get("request_rate_trend_category", "stable")
        act_trend = observation.get("action_trend_category", "stable")
        replicas = int(observation.get("current_replicas", 1))
        replica_util = float(observation.get("replica_utilization", 0.0))

        # Color coding for metrics
        cpu_col = _color(cpu, warn=70, crit=90)
        mem_col = _color(mem, warn=75, crit=90)
        rt_col = _color(rt, warn=70, crit=90)
        req_col = _color(req_rate_norm, warn=70, crit=90)

        # Progress bars
        cpu_bar = _bar(cpu)
        mem_bar = _bar(mem)
        rt_bar = _bar(rt)
        req_bar = _bar(req_rate_norm)

        # Get Q-values for current state
        state_key = agent.get_state_key(observation)
        q_vals, qmax, best_idx = _safe_q_values(agent, state_key)

        RESET = "\033[0m"

        # Line 1: Core metrics (CPU, MEM, RT, ReqRate)
        hdr = (
            f"{_ARROW} Iter {iter_no:02d} "
            if isinstance(iter_no, int)
            else f"{_ARROW} "
        )
        cpu_str = f"{cpu_col}CPU {_fmt_pct(cpu)} {cpu_bar}{RESET}"
        mem_str = f"{mem_col}MEM {_fmt_pct(mem)} {mem_bar}{RESET}"
        rt_str = f"{rt_col}RT {_fmt_pct(rt)} {rt_bar}{RESET} ({rt_ms:.1f} ms)"
        req_str = f"{req_col}REQ {_fmt_pct(req_rate_norm)} {req_bar}{RESET} ({req_rate_raw:.1f} rps)"

        logger.info(f"{hdr}| {cpu_str} | {mem_str} | {rt_str} | {req_str}")

        # Line 2: Trends, replicas, actions, Q-values
        rep_str = f"Replicas {replicas} ({_fmt_pct(replica_util)})"
        act_str = f"ACT {int(act):3d}"
        trend_str = f"Trends: Req={req_trend.upper():>6s} Act={act_trend.upper():>6s}"

        if qmax is not None and best_idx is not None:
            q_str = f"Qmax {qmax:+.3f}"
            best_s = f"Best {best_idx:3d}"
        else:
            q_str, best_s = "Qmax  n/a", "Best  n/a"

        logger.info(
            f"{'':>{len(hdr)}}| {rep_str:25s} | {act_str} | {trend_str:30s} | {q_str} | {best_s}"
        )

    except Exception as e:
        logger.warning(f"Error in verbose logging: {e}")
        logger.debug(f"Observation: {observation}")
