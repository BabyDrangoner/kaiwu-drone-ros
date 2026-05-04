"""grid_env_node — owns the GridDroneEnv, drives the closed loop.

Topics
------
out: /drone/env_obs   std_msgs/String   (JSON-encoded env_obs)
out: /drone/snapshot  std_msgs/String   (JSON-encoded env.snapshot() for viz)
in : /drone/render_ack std_msgs/Int32   (frame_no ack from viz)
in : /drone/cmd_step  std_msgs/Int8     (discrete cell step from controller)

Behaviour
~~~~~~~~~
On ``reset`` (start or after terminated/truncated) it publishes the initial
observation. After every ``cmd_step`` it advances the env and publishes a new
observation. This makes the loop strictly event-driven, no fixed dt is needed.
"""

from __future__ import annotations

import json

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_msgs.msg import String, Int8, Int32

from drone_grid_sim._bootstrap import bootstrap

bootstrap()
from closeloop.env import GridDroneEnv  # noqa: E402


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _serialize_obs(env_obs: dict) -> str:
    """env_obs already only contains json-friendly types from GridDroneEnv."""
    return _json_dumps(_to_jsonable(env_obs))


def _serialize_snapshot(snap: dict) -> str:
    return _json_dumps(_to_jsonable(snap))


class GridEnvNode(Node):
    def __init__(self) -> None:
        super().__init__("grid_env_node")

        self.declare_parameter("map_source", "official_json")
        self.declare_parameter("racer_map_name", "pillar")
        self.declare_parameter("racer_resolution", 0.5)
        self.declare_parameter("racer_height_thresh", 0.15)
        self.declare_parameter("sensor_range_cells", 10)
        self.declare_parameter("sensor_fov_deg", 140.0)
        self.declare_parameter("sensor_ray_count", 121)
        self.declare_parameter("sensor_omni_when_idle", True)
        self.declare_parameter("grid_size", 128)
        self.declare_parameter("obstacle_ratio", 0.10)
        self.declare_parameter("seed", 42)
        self.declare_parameter("episodes", 0)  # 0 = infinite
        self.declare_parameter("auto_reset_pause_sec", 0.5)
        self.declare_parameter("use_motion_controller", False)
        self.declare_parameter("action_topic", "")
        self.declare_parameter("min_action_interval_sec", 0.0)
        self.declare_parameter("sync_step_with_viz", True)
        self.declare_parameter("render_ack_topic", "/drone/render_ack")
        self.declare_parameter("step_sync_ack_timeout_sec", 0.6)
        # usr_conf overrides
        self.declare_parameter("drone_count", 4)
        self.declare_parameter("charger_count", 3)
        self.declare_parameter("station_count", 10)
        self.declare_parameter("max_step", 1000)
        self.declare_parameter("battery_max", 250)

        gp = self.get_parameter
        self.env = GridDroneEnv(
            map_source=str(gp("map_source").value),
            racer_map_name=str(gp("racer_map_name").value),
            racer_resolution=float(gp("racer_resolution").value),
            racer_height_thresh=float(gp("racer_height_thresh").value),
            grid_size=int(gp("grid_size").value),
            obstacle_ratio=float(gp("obstacle_ratio").value),
            seed=int(gp("seed").value),
        )
        self.usr_conf = {
            "map_source": str(gp("map_source").value),
            "racer_map_name": str(gp("racer_map_name").value),
            "racer_resolution": float(gp("racer_resolution").value),
            "racer_height_thresh": float(gp("racer_height_thresh").value),
            "sensor_range_cells": int(gp("sensor_range_cells").value),
            "sensor_fov_deg": float(gp("sensor_fov_deg").value),
            "sensor_ray_count": int(gp("sensor_ray_count").value),
            "sensor_omni_when_idle": bool(gp("sensor_omni_when_idle").value),
            "npc_motion_mode": "official_like",
            "drone_count": int(gp("drone_count").value),
            "charger_count": int(gp("charger_count").value),
            "station_count": int(gp("station_count").value),
            "max_step": int(gp("max_step").value),
            "battery_max": int(gp("battery_max").value),
        }
        self.episodes_target = int(gp("episodes").value)
        self.auto_reset_pause = float(gp("auto_reset_pause_sec").value)
        self.episode_idx = 0
        self.step_idx = 0
        self.total_reward = 0.0
        self._pending_reset = False
        self.min_action_interval_sec = float(gp("min_action_interval_sec").value)
        self._last_action_ns = 0
        self.sync_step_with_viz = bool(gp("sync_step_with_viz").value)
        self.render_ack_topic = str(gp("render_ack_topic").value)
        self.step_sync_ack_timeout_sec = float(gp("step_sync_ack_timeout_sec").value)
        self._waiting_render_ack = False
        self._await_frame_no = -1
        self._await_since_ns = 0
        self._queued_action: int | None = None

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Snapshot stream — every frame matters for the GIF; use a deep reliable
        # buffer instead of latched, so the viz node doesn't lose intermediate
        # frames when the loop runs fast.
        snap_qos = QoSProfile(
            depth=200,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.pub_obs = self.create_publisher(String, "/drone/env_obs", latched_qos)
        self.pub_snap = self.create_publisher(String, "/drone/snapshot", snap_qos)
        self.pub_episode = self.create_publisher(String, "/drone/episode_event", 10)

        action_topic = str(gp("action_topic").value).strip()
        use_motion_controller = bool(gp("use_motion_controller").value)
        self.action_topic = action_topic or (
            "/drone/cmd_step" if use_motion_controller else "/drone/action"
        )
        self.sub_step = self.create_subscription(
            Int8, self.action_topic, self._on_cmd_step, 10
        )
        self.sub_render_ack = self.create_subscription(
            Int32, self.render_ack_topic, self._on_render_ack, 10
        )
        self.get_logger().info(
            f"[env] subscribed action topic: {self.action_topic}, "
            f"min_action_interval_sec={self.min_action_interval_sec:.3f}, "
            f"sync_step_with_viz={self.sync_step_with_viz}, "
            f"render_ack_topic={self.render_ack_topic}"
        )

        # Kick off the first episode after the executor starts spinning.
        self._reset_timer = self.create_timer(0.05, self._maybe_kick_first_reset)

    # ------------------------------------------------------------------ #
    def _maybe_kick_first_reset(self) -> None:
        self._reset_timer.cancel()
        self._reset_episode()

    def _reset_episode(self) -> None:
        # Clear cross-episode sync/action residue before publishing new reset.
        self._waiting_render_ack = False
        self._await_frame_no = -1
        self._await_since_ns = 0
        self._queued_action = None
        self._last_action_ns = 0

        env_obs = self.env.reset(self.usr_conf)
        self.episode_idx += 1
        self.step_idx = 0
        self.total_reward = 0.0
        self.get_logger().info(
            f"[env] episode {self.episode_idx} start "
            f"(grid={self.env.grid_size}, npc={len(self.env.npcs)}, "
            f"chargers={len(self.env.chargers)}, stations={len(self.env.stations)})"
        )
        self._publish_event("reset", env_obs)
        self._publish_obs(env_obs)

    def _on_cmd_step(self, msg: Int8) -> None:
        if self._pending_reset:
            return

        now_ns = self.get_clock().now().nanoseconds
        if self.sync_step_with_viz and self._waiting_render_ack:
            # Keep only the latest action while waiting for rendering ACK.
            self._queued_action = int(msg.data)
            timeout_ns = int(max(0.0, self.step_sync_ack_timeout_sec) * 1e9)
            if timeout_ns <= 0 or self._await_since_ns <= 0:
                return
            if (now_ns - self._await_since_ns) < timeout_ns:
                return
            self.get_logger().warning(
                f"[env] render ack timeout frame={self._await_frame_no}, continue stepping"
            )
            self._waiting_render_ack = False

        self._execute_step(int(msg.data), now_ns)

    def _on_render_ack(self, msg: Int32) -> None:
        if not self.sync_step_with_viz or not self._waiting_render_ack:
            return
        frame_no = int(msg.data)
        if frame_no < self._await_frame_no:
            return
        self._waiting_render_ack = False

        # If an action arrived while we were waiting for ACK, run it now.
        if self._queued_action is not None and not self._pending_reset:
            queued = int(self._queued_action)
            self._queued_action = None
            now_ns = self.get_clock().now().nanoseconds
            self._execute_step(queued, now_ns)

    def _execute_step(self, action: int, now_ns: int) -> None:
        min_interval_ns = int(max(0.0, self.min_action_interval_sec) * 1e9)
        if min_interval_ns > 0 and self._last_action_ns > 0:
            if (now_ns - self._last_action_ns) < min_interval_ns:
                return
        if action < 0 or action > 7:
            self.get_logger().warning(f"[env] ignoring out-of-range action {action}")
            return
        self._last_action_ns = now_ns
        reward, env_obs = self.env.step(action)
        self.step_idx += 1
        self.total_reward += float(reward)
        self._publish_obs(env_obs)

        if env_obs["terminated"] or env_obs["truncated"]:
            self.get_logger().info(
                f"[env] episode {self.episode_idx} done "
                f"steps={self.step_idx} delivered={self.env.hero.delivered} "
                f"batt={self.env.hero.battery}/{self.env.hero.battery_max} "
                f"result={env_obs['extra_info']['result_message']!r} "
                f"R={self.total_reward:.2f}"
            )
            self._publish_event("done", env_obs)
            self._pending_reset = True
            self.create_timer(self.auto_reset_pause, self._auto_reset_once)

    def _auto_reset_once(self) -> None:
        if not self._pending_reset:
            return
        self._pending_reset = False
        if self.episodes_target > 0 and self.episode_idx >= self.episodes_target:
            self.get_logger().info(
                f"[env] reached target episodes={self.episodes_target}, shutting down"
            )
            self._publish_event("shutdown", {})
            rclpy.shutdown()
            return
        self._reset_episode()

    # ------------------------------------------------------------------ #
    def _publish_obs(self, env_obs: dict) -> None:
        self.pub_obs.publish(String(data=_serialize_obs(env_obs)))
        snap = self.env.snapshot()
        snap["episode"] = self.episode_idx
        self.pub_snap.publish(String(data=_serialize_snapshot(snap)))
        if self.sync_step_with_viz:
            self._await_frame_no = int(env_obs.get("frame_no", self.step_idx))
            self._await_since_ns = self.get_clock().now().nanoseconds
            self._waiting_render_ack = True

    def _publish_event(self, kind: str, env_obs: dict) -> None:
        payload = {
            "kind": kind,
            "episode": self.episode_idx,
            "step": self.step_idx,
            "delivered": self.env.hero.delivered,
            "battery": self.env.hero.battery,
            "battery_max": self.env.hero.battery_max,
            "terminated": bool(env_obs.get("terminated", False)) if env_obs else False,
            "truncated": bool(env_obs.get("truncated", False)) if env_obs else False,
            "result_message": (
                env_obs.get("extra_info", {}).get("result_message", "")
                if env_obs
                else ""
            ),
            "total_env_reward": self.total_reward,
        }
        self.pub_episode.publish(String(data=_json_dumps(payload)))


def main() -> None:
    rclpy.init()
    node = GridEnvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
