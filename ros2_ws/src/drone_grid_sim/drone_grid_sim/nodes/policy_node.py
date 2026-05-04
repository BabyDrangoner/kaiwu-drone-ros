"""policy_node — wraps agent_diy.Agent and turns env_obs JSON into a discrete action.

Topics
------
in : /drone/env_obs  std_msgs/String  (JSON env_obs from grid_env_node)
out: /drone/action   std_msgs/Int8    (0..7, raw NN output)
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String, Int8

from drone_grid_sim._bootstrap import bootstrap

bootstrap()
# Importing the agent triggers torch import, do it after bootstrap.
from agent_diy.agent import Agent  # noqa: E402


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")

        str_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        self.declare_parameter("ckpt_dir", "ckpt", descriptor=str_desc)
        self.declare_parameter("ckpt_id", "1712039", descriptor=str_desc)
        self.declare_parameter("stochastic", False)

        ckpt_dir = self.get_parameter("ckpt_dir").value
        ckpt_id = str(self.get_parameter("ckpt_id").value)
        self.stochastic = bool(self.get_parameter("stochastic").value)

        # Resolve ckpt_dir relative to KAIWU_REPO if not absolute.
        from drone_grid_sim._bootstrap import DEFAULT_REPO_ROOT
        import os
        repo = os.environ.get("KAIWU_REPO", DEFAULT_REPO_ROOT)
        if ckpt_dir and not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(repo, ckpt_dir)

        self.agent = Agent(agent_type="player", logger=self.get_logger())
        try:
            self.agent.load_model(path=ckpt_dir, id=ckpt_id)
            self.get_logger().info(
                f"[policy] loaded ckpt_dir={ckpt_dir} ckpt_id={ckpt_id}"
            )
        except Exception as exc:  # ckpt missing / shape mismatch -> warn, continue with init weights
            self.get_logger().warning(
                f"[policy] could not load ckpt ({exc}); running with random init weights"
            )

        self._last_episode_token = None  # to detect env reset and call agent.reset()
        self.pub_action = self.create_publisher(Int8, "/drone/action", 10)
        latched_sub_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, "/drone/env_obs", self._on_obs, latched_sub_qos)

    # ------------------------------------------------------------------ #
    def _on_obs(self, msg: String) -> None:
        try:
            env_obs = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"[policy] bad env_obs JSON: {exc}")
            return

        # Detect episode boundary using env step_no rolling back to 0.
        step_no = env_obs.get("observation", {}).get("step_no", 0)
        if step_no == 0:
            self.agent.reset(env_obs)
            self.get_logger().info("[policy] agent reset for new episode")

        # On terminal observation, env will reset itself; don't act.
        if env_obs.get("terminated") or env_obs.get("truncated"):
            return

        try:
            obs_data, _ = self.agent.observation_process(env_obs)
            act_data = self.agent.predict(list_obs_data=[obs_data])[0]
            action = self.agent.action_process(act_data, is_stochastic=self.stochastic)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"[policy] inference failed: {exc!r}")
            return

        self.pub_action.publish(Int8(data=int(action)))


def main() -> None:
    rclpy.init()
    node = PolicyNode()
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
