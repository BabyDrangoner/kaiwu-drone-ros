"""Minimal stand-ins for the kaiwudrl / common_python / tools packages
agent_diy depends on, so the agent can run outside the official framework.

Importing this module installs the stubs into ``sys.modules`` so subsequent
``import kaiwudrl...`` etc. resolve to the implementations defined here.
"""

from __future__ import annotations

import sys
import types
from collections import namedtuple
from typing import Any, Dict


# ---------------------------------------------------------------------------- #
# kaiwudrl.interface.agent.BaseAgent
# ---------------------------------------------------------------------------- #
class _BaseAgent:
    """No-op base class. The real framework provides device/logger/monitor
    plumbing; for our closed loop the subclass already stores them itself."""

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        # agent_diy.Agent.__init__ already saved everything it needs before
        # calling super().__init__(); we just have to be a benign target.
        self.agent_type = agent_type
        if not hasattr(self, "device") or self.device is None:
            self.device = device
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logger
        if not hasattr(self, "monitor") or self.monitor is None:
            self.monitor = monitor

    # Methods the workflow may call generically; they are normally overridden.
    def reset(self, env_obs=None):
        return None

    def predict(self, list_obs_data):
        raise NotImplementedError

    def learn(self, list_sample_data):
        return None

    def save_model(self, path=None, id="1"):
        return None

    def load_model(self, path=None, id="1"):
        return None

    def send_sample_data(self, samples):  # pragma: no cover - optional hook
        return None


# ---------------------------------------------------------------------------- #
# common_python.utils.common_func.create_cls
# ---------------------------------------------------------------------------- #
def create_cls(name: str, **fields) -> type:
    """Lightweight ``namedtuple``-style factory used by agent_diy for ObsData /
    ActData / SampleData. Accepts arbitrary keyword defaults."""

    field_names = list(fields.keys())
    defaults = tuple(fields.values())
    cls = namedtuple(name, field_names, defaults=defaults)
    return cls


# ---------------------------------------------------------------------------- #
# common_python.utils.workflow_disaster_recovery.handle_disaster_recovery
# ---------------------------------------------------------------------------- #
def handle_disaster_recovery(env_obs: Any, logger=None) -> bool:
    """In the official framework this detects unrecoverable env errors. For
    our local loop we treat ``None`` / missing observation as a soft skip and
    everything else as healthy."""

    if env_obs is None:
        if logger:
            logger.warning("env_obs is None, skipping episode")
        return True
    if isinstance(env_obs, dict) and env_obs.get("__disaster__"):
        if logger:
            logger.warning("env_obs flagged as disaster, skipping episode")
        return True
    return False


# ---------------------------------------------------------------------------- #
# tools.metrics_utils.get_training_metrics
# ---------------------------------------------------------------------------- #
def get_training_metrics() -> Dict[str, Any]:
    """Stub: the official framework returns runtime KPIs (FPS, queue depth,
    etc.). We return an empty dict; the workflow only logs it when truthy."""

    return {}


# ---------------------------------------------------------------------------- #
# tools.train_env_conf_validate.read_usr_conf
# ---------------------------------------------------------------------------- #
def read_usr_conf(toml_path: str, logger=None):
    """Read the user TOML config. Falls back to the Python ``tomllib`` /
    ``tomli`` backends. Returns the dict under ``[env_conf]`` if present, else
    the whole file."""

    import os

    if not os.path.exists(toml_path):
        if logger:
            logger.error(f"usr_conf {toml_path} not found")
        return None

    try:
        try:
            import tomllib  # py>=3.11
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        if logger:
            logger.error(f"failed to parse {toml_path}: {exc}")
        return None

    return data.get("env_conf", data)


# ---------------------------------------------------------------------------- #
# Module installation
# ---------------------------------------------------------------------------- #
def _install_module(dotted_name: str, **attrs) -> types.ModuleType:
    """Create (or fetch) ``dotted_name`` in ``sys.modules`` and set attrs."""

    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            mod.__path__ = []  # mark as package so children can be added
            sys.modules[sub] = mod
        # link parent.child attribute
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            setattr(parent, parts[i - 1], sys.modules[sub])
    target = sys.modules[dotted_name]
    for k, v in attrs.items():
        setattr(target, k, v)
    return target


def install() -> None:
    """Install all stub modules into ``sys.modules``. Idempotent."""

    if getattr(install, "_done", False):
        return

    _install_module("kaiwudrl")
    _install_module("kaiwudrl.interface")
    _install_module("kaiwudrl.interface.agent", BaseAgent=_BaseAgent)

    _install_module("common_python")
    _install_module("common_python.utils")
    _install_module(
        "common_python.utils.common_func", create_cls=create_cls
    )
    _install_module(
        "common_python.utils.workflow_disaster_recovery",
        handle_disaster_recovery=handle_disaster_recovery,
    )

    _install_module("tools")
    _install_module(
        "tools.metrics_utils", get_training_metrics=get_training_metrics
    )
    _install_module(
        "tools.train_env_conf_validate", read_usr_conf=read_usr_conf
    )

    install._done = True  # type: ignore[attr-defined]


# Auto-install on import for convenience.
install()
