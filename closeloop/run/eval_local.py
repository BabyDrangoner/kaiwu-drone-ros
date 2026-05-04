"""Standalone closed-loop runner: agent_diy + GridDroneEnv + GIF renderer.

This is the *non-ROS* sanity loop. It also acts as the reference implementation
for the ROS bridge — the two share the same env / renderer / agent objects.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# Ensure repo root is importable & install framework stubs first.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from closeloop import framework_stub  # noqa: F401  (auto-installs sys.modules)
from closeloop.env import GridDroneEnv
from closeloop.render import FrameRecorder


def build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("closeloop")


def load_agent(ckpt_dir: Optional[str], ckpt_id: Optional[str], logger):
    """Import agent_diy lazily so framework stubs are already in place."""

    from agent_diy.agent import Agent  # noqa: WPS433

    agent = Agent(agent_type="player", logger=logger)
    if ckpt_dir and ckpt_id:
        agent.load_model(path=ckpt_dir, id=ckpt_id)
    return agent


def run_episode(env, agent, recorder: Optional[FrameRecorder], logger, max_steps_override=None) -> dict:
    env_obs = env.reset(usr_conf=None)
    agent.reset(env_obs)

    if recorder is not None:
        recorder.render(env.snapshot(), reward_so_far=0.0)

    done = False
    total_reward = 0.0
    step = 0
    while not done:
        action = agent.exploit(env_obs)
        reward, env_obs = env.step(action)
        total_reward += reward
        step += 1
        if recorder is not None:
            recorder.render(env.snapshot(), reward_so_far=total_reward)
        done = bool(env_obs["terminated"] or env_obs["truncated"])
        if max_steps_override is not None and step >= max_steps_override:
            break

    pp = agent.preprocessor
    summary = {
        "steps": step,
        "delivered": pp.delivered,
        "battery_left": pp.battery,
        "battery_max": pp.battery_max,
        "terminated": env_obs["terminated"],
        "truncated": env_obs["truncated"],
        "result_message": env_obs["extra_info"].get("result_message", ""),
        "total_env_reward": total_reward,
    }
    logger.info(f"[EPISODE] {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="ckpt", help="path containing model.ckpt-<id>.pkl")
    parser.add_argument("--ckpt-id", default="1712039")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="hard cap; default uses env max_step")
    parser.add_argument("--out-dir", default="closeloop/out")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = build_logger()
    env = GridDroneEnv(seed=args.seed)
    agent = load_agent(args.ckpt_dir, args.ckpt_id, logger)

    for ep in range(args.episodes):
        recorder = None if args.no_gif else FrameRecorder(args.out_dir, episode_id=ep, fps=args.fps)
        run_episode(env, agent, recorder, logger, max_steps_override=args.max_steps)
        if recorder is not None:
            gif = recorder.save_gif()
            png = recorder.save_last_png()
            logger.info(f"[GIF] {gif}  [PNG] {png}")


if __name__ == "__main__":
    main()
