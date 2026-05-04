"""Bring up the full closed loop: env + policy + controller + viz.

Optional ``rviz:=true`` arg launches RViz with the bundled config.
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("drone_grid_sim")
    rviz_cfg = os.path.join(pkg_share, "rviz", "drone_grid_sim.rviz")

    args = [
        DeclareLaunchArgument("ckpt_dir", default_value="ckpt"),
        DeclareLaunchArgument("ckpt_id", default_value="1712039"),
        DeclareLaunchArgument("episodes", default_value="1"),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("map_source", default_value="official_json"),
        DeclareLaunchArgument("racer_map_name", default_value="pillar"),
        DeclareLaunchArgument("racer_resolution", default_value="0.5"),
        DeclareLaunchArgument("racer_height_thresh", default_value="0.15"),
        DeclareLaunchArgument("sensor_range_cells", default_value="10"),
        DeclareLaunchArgument("sensor_fov_deg", default_value="140.0"),
        DeclareLaunchArgument("sensor_ray_count", default_value="121"),
        DeclareLaunchArgument("sensor_omni_when_idle", default_value="true"),
        # Official baseline map shape
        DeclareLaunchArgument("grid_size",       default_value="128"),
        DeclareLaunchArgument("drone_count", default_value="4"),
        DeclareLaunchArgument("charger_count",   default_value="3"),
        DeclareLaunchArgument("station_count",   default_value="10"),
        DeclareLaunchArgument("max_step",         default_value="1000"),
        DeclareLaunchArgument("battery_max",      default_value="250"),
        DeclareLaunchArgument("step_seconds",     default_value="0.12"),
        DeclareLaunchArgument("control_rate_hz",  default_value="30.0"),
        DeclareLaunchArgument("throttle_step",    default_value="true"),
        DeclareLaunchArgument("anti_oscillation", default_value="true"),
        DeclareLaunchArgument("use_motion_controller", default_value="false"),
        # Leave empty to auto-select based on use_motion_controller:
        # false -> /drone/action, true -> /drone/cmd_step
        DeclareLaunchArgument("action_topic", default_value=""),
        # 0 means no drop-throttling; the loop is already observation-driven.
        DeclareLaunchArgument("min_action_interval_sec", default_value="0.0"),
        DeclareLaunchArgument("sync_step_with_viz", default_value="true"),
        DeclareLaunchArgument("render_ack_topic", default_value="/drone/render_ack"),
        DeclareLaunchArgument("step_sync_ack_timeout_sec", default_value="0.6"),
        DeclareLaunchArgument("record_gif", default_value="false"),
        DeclareLaunchArgument("gif_out_dir", default_value="ros2_ws/gif_out"),
        DeclareLaunchArgument("show_all_elements", default_value="false"),
        DeclareLaunchArgument("motion_preview_enabled", default_value="true"),
        DeclareLaunchArgument("motion_preview_steps", default_value="10"),
        DeclareLaunchArgument("motion_preview_accel", default_value="0.35"),
        DeclareLaunchArgument("motion_preview_decel", default_value="0.45"),
        DeclareLaunchArgument("motion_preview_max_speed", default_value="1.4"),
        DeclareLaunchArgument("motion_preview_turn_blend", default_value="0.40"),
        # RACER-style frame_id
        DeclareLaunchArgument("frame_id", default_value="map"),
        # rviz:=true  launches RViz2 for interactive 2D city-grid view.
        # Requires a display: WSLg (Win 11) sets DISPLAY automatically;
        # Win 10 users need VcXsrv and set DISPLAY=:0 beforehand.
        DeclareLaunchArgument("rviz", default_value="true"),
    ]

    env_node = Node(
        package="drone_grid_sim",
        executable="grid_env_node",
        name="grid_env_node",
        output="screen",
        parameters=[{
            "map_source": LaunchConfiguration("map_source"),
            "racer_map_name": LaunchConfiguration("racer_map_name"),
            "racer_resolution": LaunchConfiguration("racer_resolution"),
            "racer_height_thresh": LaunchConfiguration("racer_height_thresh"),
            "sensor_range_cells": LaunchConfiguration("sensor_range_cells"),
            "sensor_fov_deg": LaunchConfiguration("sensor_fov_deg"),
            "sensor_ray_count": LaunchConfiguration("sensor_ray_count"),
            "sensor_omni_when_idle": LaunchConfiguration("sensor_omni_when_idle"),
            "grid_size": LaunchConfiguration("grid_size"),
            "seed": LaunchConfiguration("seed"),
            "episodes": LaunchConfiguration("episodes"),
            "drone_count": LaunchConfiguration("drone_count"),
            "charger_count": LaunchConfiguration("charger_count"),
            "station_count": LaunchConfiguration("station_count"),
            "max_step": LaunchConfiguration("max_step"),
            "battery_max": LaunchConfiguration("battery_max"),
            "use_motion_controller": ParameterValue(LaunchConfiguration("use_motion_controller"), value_type=bool),
            "action_topic": LaunchConfiguration("action_topic"),
            "min_action_interval_sec": LaunchConfiguration("min_action_interval_sec"),
            "sync_step_with_viz": ParameterValue(LaunchConfiguration("sync_step_with_viz"), value_type=bool),
            "render_ack_topic": LaunchConfiguration("render_ack_topic"),
            "step_sync_ack_timeout_sec": LaunchConfiguration("step_sync_ack_timeout_sec"),
        }],
    )

    policy = ExecuteProcess(
        cmd=[
            "python",
            "-m",
            "drone_grid_sim.nodes.policy_node",
            "--ros-args",
            "-r",
            "__node:=policy_node",
            "-p",
            ["ckpt_dir:=", LaunchConfiguration("ckpt_dir")],
            "-p",
            "stochastic:=false",
        ],
        output="screen",
    )

    ctrl = Node(
        package="drone_grid_sim",
        executable="motion_controller_node",
        name="motion_controller_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_motion_controller")),
        parameters=[{
            "step_seconds": LaunchConfiguration("step_seconds"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "throttle_step": ParameterValue(LaunchConfiguration("throttle_step"), value_type=bool),
            "anti_oscillation": ParameterValue(LaunchConfiguration("anti_oscillation"), value_type=bool),
        }],
    )

    viz = Node(
        package="drone_grid_sim",
        executable="viz_node",
        name="viz_node",
        output="screen",
        parameters=[{
            "frame_id": ParameterValue(LaunchConfiguration("frame_id"), value_type=str),
            "record_gif": ParameterValue(LaunchConfiguration("record_gif"), value_type=bool),
            "gif_out_dir": ParameterValue(LaunchConfiguration("gif_out_dir"), value_type=str),
            "show_all_elements": ParameterValue(LaunchConfiguration("show_all_elements"), value_type=bool),
            "render_ack_topic": ParameterValue(LaunchConfiguration("render_ack_topic"), value_type=str),
            "motion_interp_step_seconds": LaunchConfiguration("step_seconds"),
            "motion_preview_enabled": ParameterValue(LaunchConfiguration("motion_preview_enabled"), value_type=bool),
            "motion_preview_steps": LaunchConfiguration("motion_preview_steps"),
            "motion_preview_accel": LaunchConfiguration("motion_preview_accel"),
            "motion_preview_decel": LaunchConfiguration("motion_preview_decel"),
            "motion_preview_max_speed": LaunchConfiguration("motion_preview_max_speed"),
            "motion_preview_turn_blend": LaunchConfiguration("motion_preview_turn_blend"),
        }],
    )

    # Forward the host DISPLAY into rviz2 so it renders in WSLg / X11.
    import os as _os
    _display = _os.environ.get("DISPLAY", ":0")
    rviz = ExecuteProcess(
        cmd=["rviz2", "-d", rviz_cfg],
        additional_env={"DISPLAY": _display},
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([*args, env_node, policy, ctrl, viz, rviz])
