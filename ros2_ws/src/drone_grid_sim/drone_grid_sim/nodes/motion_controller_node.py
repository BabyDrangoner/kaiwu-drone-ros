"""motion_controller_node — turn the policy's discrete action id into a
controller-style command (Twist) **and** feed the env one cell step.

Why both?
~~~~~~~~~
The user wants the network output to be *executed by a motion controller*.
We emit a `geometry_msgs/Twist` with cell-per-second velocity (so a real
drone/car node could subscribe and integrate), and at the same time emit a
discrete `Int8` cell-step command that closes the loop with grid_env_node.

Topics
------
in : /drone/action     std_msgs/Int8       (raw policy output 0..7)
out: /drone/cmd_vel    geometry_msgs/Twist (continuous-style controller cmd)
out: /drone/cmd_step   std_msgs/Int8       (discrete step echoed to env)
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist


# Mirror closeloop.env.grid_env.ACTION_DELTA — keep in sync.
ACTION_DELTA = (
    (1, 0),    # 0 right
    (1, -1),   # 1 right-up
    (0, -1),   # 2 up
    (-1, -1),  # 3 left-up
    (-1, 0),   # 4 left
    (-1, 1),   # 5 left-down
    (0, 1),    # 6 down
    (1, 1),    # 7 right-down
)


class MotionControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_controller_node")
        self.declare_parameter("step_seconds", 0.2)  # time to traverse one cell
        self.declare_parameter("publish_twist", True)
        self.declare_parameter("throttle_step", True)
        self.declare_parameter("anti_oscillation", True)

        self.step_seconds = float(self.get_parameter("step_seconds").value)
        self.publish_twist = bool(self.get_parameter("publish_twist").value)
        self.throttle_step = bool(self.get_parameter("throttle_step").value)
        self.anti_oscillation = bool(self.get_parameter("anti_oscillation").value)

        self.pub_step = self.create_publisher(Int8, "/drone/cmd_step", 10)
        self.pub_vel = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.create_subscription(Int8, "/drone/action", self._on_action, 10)

        self._pending_action: int | None = None
        self._last_emit_ns: int = 0
        self._last_emitted_action: int | None = None
        self._emit_period_ns: int = max(int(self.step_seconds * 1e9), int(1e7))
        self._timer = self.create_timer(max(self.step_seconds, 0.01), self._on_tick)
        self.get_logger().info(
            f"[ctrl] throttle_step={self.throttle_step} anti_oscillation={self.anti_oscillation} step_seconds={self.step_seconds:.3f}"
        )

    def _on_action(self, msg: Int8) -> None:
        action = int(msg.data)
        if action < 0 or action >= len(ACTION_DELTA):
            self.get_logger().warning(f"[ctrl] dropping out-of-range action {action}")
            return

        self._pending_action = action
        if not self.throttle_step:
            self._emit_step(action)

    def _emit_step(self, action: int) -> None:
        if (
            self.anti_oscillation
            and self._last_emitted_action is not None
            and action == self._opposite_action(self._last_emitted_action)
        ):
            # Keep previous heading for one more tick to break A<->B ping-pong.
            action = self._last_emitted_action

        if self.publish_twist:
            dx, dz = ACTION_DELTA[action]
            tw = Twist()
            tw.linear.x = dx / max(self.step_seconds, 1e-3)
            tw.linear.y = -dz / max(self.step_seconds, 1e-3)  # ROS y is north
            tw.angular.z = math.atan2(-dz, dx)
            self.pub_vel.publish(tw)
        self.pub_step.publish(Int8(data=action))
        self._last_emitted_action = action

    @staticmethod
    def _opposite_action(action: int) -> int:
        # 0<->4, 1<->5, 2<->6, 3<->7
        return (action + 4) % len(ACTION_DELTA)

    def _on_tick(self) -> None:
        if not self.throttle_step:
            return
        if self._pending_action is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._last_emit_ns and (now_ns - self._last_emit_ns) < self._emit_period_ns:
            return
        action = int(self._pending_action)
        self._pending_action = None
        self._emit_step(action)
        self._last_emit_ns = now_ns


def main() -> None:
    rclpy.init()
    node = MotionControllerNode()
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
