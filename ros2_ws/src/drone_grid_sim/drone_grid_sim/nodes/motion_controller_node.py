"""motion_controller_node — execute policy action as continuous motion first.

In strict mode (throttle_step=true), a policy action starts a continuous
velocity command for one cell duration, and only after the motion window is
finished do we emit one /drone/cmd_step to advance the grid env.

This enforces: one policy decision -> one destination reached -> next policy.

Topics
------
in : /drone/action     std_msgs/Int8       (raw policy output 0..7)
out: /drone/cmd_vel    geometry_msgs/Twist (continuous controller command)
out: /drone/cmd_step   std_msgs/Int8       (step committed when arrived)
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
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("publish_twist", True)
        self.declare_parameter("throttle_step", True)
        self.declare_parameter("anti_oscillation", True)

        self.step_seconds = float(self.get_parameter("step_seconds").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.publish_twist = bool(self.get_parameter("publish_twist").value)
        self.throttle_step = bool(self.get_parameter("throttle_step").value)
        self.anti_oscillation = bool(self.get_parameter("anti_oscillation").value)

        self.pub_step = self.create_publisher(Int8, "/drone/cmd_step", 10)
        self.pub_vel = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.create_subscription(Int8, "/drone/action", self._on_action, 10)

        self._pending_action: int | None = None
        self._last_emitted_action: int | None = None
        self._active_action: int | None = None
        self._motion_deadline_ns: int = 0
        self._motion_window_ns: int = max(int(self.step_seconds * 1e9), int(1e7))
        timer_period = 1.0 / max(self.control_rate_hz, 1.0)
        self._timer = self.create_timer(timer_period, self._on_tick)
        self.get_logger().info(
            f"[ctrl] throttle_step={self.throttle_step} anti_oscillation={self.anti_oscillation} "
            f"step_seconds={self.step_seconds:.3f} control_rate_hz={self.control_rate_hz:.1f}"
        )

    def _on_action(self, msg: Int8) -> None:
        action = int(msg.data)
        if action < 0 or action >= len(ACTION_DELTA):
            self.get_logger().warning(f"[ctrl] dropping out-of-range action {action}")
            return

        if not self.throttle_step:
            self._emit_step(action)
            return

        # In strict arrival mode, reject/replace actions while one motion is in flight.
        if self._active_action is not None:
            self._pending_action = action
            return

        self._start_motion(action)

    def _start_motion(self, action: int) -> None:
        action = self._sanitize_action(action)
        self._active_action = action
        now_ns = self.get_clock().now().nanoseconds
        self._motion_deadline_ns = now_ns + self._motion_window_ns
        if self.publish_twist:
            self.pub_vel.publish(self._twist_for_action(action))

    def _emit_step(self, action: int) -> None:
        action = self._sanitize_action(action)

        if self.publish_twist:
            self.pub_vel.publish(self._twist_for_action(action))
        self.pub_step.publish(Int8(data=action))
        self._last_emitted_action = action

    def _sanitize_action(self, action: int) -> int:
        if (
            self.anti_oscillation
            and self._last_emitted_action is not None
            and action == self._opposite_action(self._last_emitted_action)
        ):
            # Keep previous heading for one more tick to break A<->B ping-pong.
            return self._last_emitted_action
        return action

    def _twist_for_action(self, action: int) -> Twist:
        dx, dz = ACTION_DELTA[action]
        tw = Twist()
        tw.linear.x = dx / max(self.step_seconds, 1e-3)
        tw.linear.y = -dz / max(self.step_seconds, 1e-3)  # ROS y is north
        tw.angular.z = math.atan2(-dz, dx)
        return tw

    def _publish_stop(self) -> None:
        if not self.publish_twist:
            return
        self.pub_vel.publish(Twist())

    @staticmethod
    def _opposite_action(action: int) -> int:
        # 0<->4, 1<->5, 2<->6, 3<->7
        return (action + 4) % len(ACTION_DELTA)

    def _on_tick(self) -> None:
        if not self.throttle_step:
            return

        if self._active_action is None:
            if self._pending_action is None:
                return
            pending = int(self._pending_action)
            self._pending_action = None
            self._start_motion(pending)
            return

        now_ns = self.get_clock().now().nanoseconds
        action = int(self._active_action)

        # Keep sending velocity while moving to destination.
        if now_ns < self._motion_deadline_ns:
            if self.publish_twist:
                self.pub_vel.publish(self._twist_for_action(action))
            return

        # Arrival reached -> stop continuous command and commit one env step.
        self._publish_stop()
        self.pub_step.publish(Int8(data=action))
        self._last_emitted_action = action
        self._active_action = None

        # Drop pre-arrival queued actions to force a fresh policy decision
        # from the newly reached state.
        self._pending_action = None


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
