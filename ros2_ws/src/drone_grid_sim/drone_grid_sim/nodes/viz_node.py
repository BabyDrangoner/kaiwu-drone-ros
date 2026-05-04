"""viz_node — RACER-style city-grid RViz2 visualization.

RACER-style rendering with PointCloud2 AxisColor-Z rainbow for occupancy map,
plus MarkerArray for dynamic entities (drone, NPCs, stations, trail).

Topics:
  /drone/occupancy_cloud   PointCloud2   obstacle cells with Z-axis rainbow color
  /drone/unknown_cloud     PointCloud2   unexplored cells (flat grey)
  /drone/markers_static    MarkerArray   ground, roads, buildings, landmarks (transient local)
  /drone/markers_dynamic   MarkerArray   drone quad, NPCs, trail, station highlights
  /drone/drone_marker      Marker        single drone marker for trajectory group
  /drone/trail_marker      Marker        trail line strip
"""

from __future__ import annotations

import collections
import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, String, Int32
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

from drone_grid_sim._bootstrap import bootstrap

bootstrap()
from closeloop.env.official_map import load_cell_type_palette  # noqa: E402
from closeloop.env.racer_map import (  # noqa: E402
    filter_point_cloud_by_explored,
    height_map_to_point_cloud,
)


# ─── colour palette ───────────────────────────────────────────────────────────
def _c(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


C_GROUND    = _c(.62, .62, .58)
C_ROAD      = _c(.48, .48, .45, .90)
C_WAREHOUSE = _c(.10, .50, .90)
C_CHARGER   = _c(.98, .68, .04)
C_STATION   = _c(.15, .75, .25)
C_TARGET    = _c(.95, .15, .10)
C_NPC       = _c(.70, .10, .85)
C_NPC_ZONE  = _c(.80, .20, .82, .12)
C_DRONE     = _c(.98, .35, .05)
C_DRONE_ARM = _c(.22, .22, .22)
C_WHITE     = _c(1., 1., 1.)
C_YELLOW    = _c(1., 1., .20)
C_TRAIL     = _c(.98, .58, .0, .75)
C_GREEN_BAR = _c(.0,  .85, .20)
C_AMBER_BAR = _c(.92, .72, .0)
C_RED_BAR   = _c(.92, .12, .12)
C_MAP_GRAY  = _c(.55, .55, .55, .55)

# RACER-style building height range for Z-axis coloring
BUILDING_Z_MIN = 0.0
BUILDING_Z_MAX = 3.5


# ─── building helpers ─────────────────────────────────────────────────────────
def _building_height(x: int, z: int) -> float:
    """Deterministic per-cell height in [2.0, 7.5] m (fallback for city maps)."""
    return 2.0 + ((x * 13 + z * 7 + (x ^ z) * 3) % 12) * 0.5


def _height_to_z(h: float) -> float:
    """Map real height to Z value for AxisColor-Z rainbow."""
    return BUILDING_Z_MIN + max(0.0, h - 2.0) / 5.5 * (BUILDING_Z_MAX - BUILDING_Z_MIN)


def _bldg_color_from_height(h: float) -> ColorRGBA:
    """HSV rainbow by height: short=blue, medium=green/yellow, tall=red."""
    t = max(0.0, min(1.0, (h - 2.0) / 5.5))
    hue = 240.0 * (1.0 - t)
    s, v = 0.82, 0.88
    h60 = hue / 60.0
    i = int(h60) % 6
    f = h60 - int(h60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    u = v * (1 - (1 - f) * s)
    r, g, b = [(v, u, p), (q, v, p), (p, v, u),
               (p, q, v), (u, p, v), (v, p, q)][i]
    return _c(r, g, b)


def _bldg_color(x: int, z: int) -> ColorRGBA:
    """HSV rainbow by deterministic height (fallback)."""
    return _bldg_color_from_height(_building_height(x, z))


# ─── PointCloud2 helpers ──────────────────────────────────────────────────────
def _make_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Create a PointCloud2 message from Nx3 float32 array (x, y, z)."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * points.shape[0]
    msg.is_dense = True
    # Add intensity column (same as z for AxisColor-Z rainbow)
    intensity = points[:, 2:3]
    data = np.hstack([points, intensity]).astype(np.float32)
    msg.data = data.tobytes()
    return msg


def _build_explored_obstacle_cloud(
    points: np.ndarray,
    frame_id: str, stamp,
) -> PointCloud2:
    """Build PointCloud2 for explored obstacle points."""

    if points.size == 0:
        points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    ros_points = np.empty_like(points)
    ros_points[:, 0] = points[:, 0]
    ros_points[:, 1] = -points[:, 1]
    ros_points[:, 2] = points[:, 2]
    return _make_pointcloud2(ros_points, frame_id, stamp)


def _build_unexplored_cloud(
    grid: List[List[int]],
    explored_mask: List[List[int]],
    frame_id: str, stamp,
) -> PointCloud2:
    """Build PointCloud2 for unexplored cells (flat grey fog)."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    pts = []
    for zi in range(H):
        for xi in range(W):
            if not explored_mask[zi][xi]:
                pts.append([float(xi), -float(zi), -0.5])
    if not pts:
        pts = [[0.0, 0.0, -0.5]]
    return _make_pointcloud2(np.array(pts, dtype=np.float32), frame_id, stamp)


def _build_ground_truth_cloud(
    grid: List[List[int]],
    height_map: Optional[List[List[float]]],
    frame_id: str, stamp,
) -> PointCloud2:
    """Build faint ground-truth cloud for spatial reference (RACER simulation_map style)."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    pts = []
    for zi in range(H):
        for xi in range(W):
            if grid[zi][xi]:
                if height_map is not None:
                    h = float(height_map[zi][xi])
                    if h <= 0.0:
                        h = _building_height(xi, zi)
                else:
                    h = _building_height(xi, zi)
                pts.append([float(xi), -float(zi), _height_to_z(h)])
            else:
                pts.append([float(xi), -float(zi), -0.3])
    if not pts:
        pts = [[0.0, 0.0, 0.0]]
    return _make_pointcloud2(np.array(pts, dtype=np.float32), frame_id, stamp)


# ─── marker helpers ───────────────────────────────────────────────────────────
# Keep dynamic markers alive long enough for slow/heavy frames.
# A short lifetime causes visible blinking when frame intervals fluctuate.
_DYN_LIFE_NS = int(20.0e9)


def _mk(ns: str, mid: int, mtype: int, fid: str, stamp, life_ns: int = 0) -> Marker:
    m = Marker()
    m.header.frame_id = fid
    m.header.stamp = stamp
    m.ns = ns; m.id = mid; m.type = mtype; m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    if life_ns:
        m.lifetime.nanosec = life_ns
    return m


def _pt(x: float, y: float, z: float = 0.0) -> Point:
    return Point(x=float(x), y=float(y), z=float(z))


def _sc(x: float, y: float, z: float) -> Vector3:
    return Vector3(x=float(x), y=float(y), z=float(z))


# ─── static markers ───────────────────────────────────────────────────────────
def mk_ground(W: int, H: int, stamp, fid: str) -> Marker:
    m = _mk("ground", 0, Marker.CUBE, fid, stamp)
    m.pose.position = _pt((W - 1) / 2.0, -(H - 1) / 2.0, -0.08)
    m.scale = _sc(float(W) + 1.0, float(H) + 1.0, 0.14)
    m.color = C_GROUND
    return m


def mk_road_grid(W: int, H: int, stamp, fid: str) -> Marker:
    m = _mk("roads", 0, Marker.LINE_LIST, fid, stamp)
    m.scale.x = 0.18
    m.color = C_ROAD
    step = 9 if max(W, H) > 64 else 1
    for xi in range(0, W + 1, step):
        m.points += [_pt(xi - 0.5, 0.5, 0.02), _pt(xi - 0.5, -(H - 0.5), 0.02)]
    for zi in range(0, H + 1, step):
        m.points += [_pt(-0.5, -(zi - 0.5), 0.02), _pt(W - 0.5, -(zi - 0.5), 0.02)]
    return m


def mk_building(x: int, z: int, mid: int, stamp, fid: str) -> Marker:
    h = _building_height(x, z)
    m = _mk("bldg", mid, Marker.CUBE, fid, stamp)
    m.pose.position = _pt(float(x), -float(z), h / 2.0)
    m.scale = _sc(0.88, 0.88, h)
    m.color = _bldg_color(x, z)
    return m


def mk_building_list(grid: List[List[int]], stamp, fid: str) -> Marker:
    m = _mk("bldg", 0, Marker.CUBE_LIST, fid, stamp)
    m.scale = _sc(0.88, 0.88, 0.88)
    for zi, row in enumerate(grid):
        for xi, cell in enumerate(row):
            if not cell:
                continue
            h = _building_height(xi, zi)
            # CUBE_LIST uses a uniform scale, so stack cubes vertically.
            levels = max(1, int(round(h / m.scale.z)))
            color = _bldg_color(xi, zi)
            for level in range(levels):
                m.points.append(_pt(float(xi), -float(zi), m.scale.z * (0.5 + level)))
                m.colors.append(color)
    return m


def _cell_is_explored(explored_mask: Optional[List[List[int]]], pos: Tuple[int, int]) -> bool:
    if explored_mask is None:
        return True
    x, z = int(pos[0]), int(pos[1])
    H = len(explored_mask)
    W = len(explored_mask[0]) if H > 0 else 0
    return 0 <= x < W and 0 <= z < H and bool(explored_mask[z][x])


_CELL_TYPE_PALETTE = load_cell_type_palette()


def _grayize(color: ColorRGBA, alpha: float = 0.55) -> ColorRGBA:
    lum = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
    gray = 0.30 + 0.55 * lum
    return _c(gray, gray, gray, alpha)


def _cell_color(cell_types: Optional[List[List[int]]], grid: List[List[int]], x: int, z: int) -> ColorRGBA:
    if isinstance(cell_types, str):
        cell_types = None
    if cell_types is not None and 0 <= z < len(cell_types) and 0 <= x < len(cell_types[z]):
        rgba = _CELL_TYPE_PALETTE.get(int(cell_types[z][x]))
        if rgba is not None:
            return _c(rgba[0], rgba[1], rgba[2], rgba[3])
    return C_GROUND if not grid[z][x] else _c(.38, .38, .38, .90)


def mk_map_cells(
    grid: List[List[int]],
    explored_mask: Optional[List[List[int]]],
    cell_types: Optional[List[List[int]]],
    stamp,
    fid: str,
    explored_only: bool,
) -> Marker:
    mid = 901 if explored_only else 900
    ns = "map_cells_explored" if explored_only else "map_cells_gray"
    marker = _mk(ns, mid, Marker.CUBE_LIST, fid, stamp, 0)
    marker.scale = _sc(0.96, 0.96, 0.04)
    for z, row in enumerate(grid):
        for x, _cell in enumerate(row):
            is_explored = _cell_is_explored(explored_mask, (x, z))
            if explored_only and not is_explored:
                continue
            if not explored_only and is_explored:
                continue
            marker.points.append(_pt(float(x), -float(z), -0.02 if explored_only else -0.04))
            base_color = _cell_color(cell_types, grid, x, z)
            marker.colors.append(base_color if explored_only else _grayize(base_color))
    return marker


def mk_warehouse(x: int, z: int, mid: int, stamp, fid: str) -> List[Marker]:
    body = _mk("wh_body", mid * 2, Marker.CUBE, fid, stamp)
    body.pose.position = _pt(float(x), -float(z), 2.6)
    body.scale = _sc(1.00, 1.00, 5.2)
    body.color = C_WAREHOUSE
    lbl = _mk("wh_lbl", mid * 2 + 1, Marker.TEXT_VIEW_FACING, fid, stamp)
    lbl.pose.position = _pt(float(x), -float(z), 5.9)
    lbl.scale.z = 1.05
    lbl.color = C_WHITE
    lbl.text = "W"
    halo = _mk("wh_halo", mid, Marker.CYLINDER, fid, stamp)
    halo.pose.position = _pt(float(x), -float(z), 0.03)
    halo.scale = _sc(1.55, 1.55, 0.06)
    halo.color = _c(C_WAREHOUSE.r, C_WAREHOUSE.g, C_WAREHOUSE.b, 0.22)
    return [body, lbl, halo]


def _gray_marker(marker: Marker, alpha: float = 0.55) -> None:
    marker.color = _grayize(marker.color, alpha)


def _gray_markers(markers: List[Marker], alpha: float = 0.55) -> List[Marker]:
    for m in markers:
        _gray_marker(m, alpha)
    return markers


def mk_charger(x: int, z: int, mid: int, stamp, fid: str) -> List[Marker]:
    cyl = _mk("ch_cyl", mid * 2, Marker.CYLINDER, fid, stamp)
    cyl.pose.position = _pt(float(x), -float(z), 1.6)
    cyl.scale = _sc(0.62, 0.62, 3.2)
    cyl.color = C_CHARGER
    lbl = _mk("ch_lbl", mid * 2 + 1, Marker.TEXT_VIEW_FACING, fid, stamp)
    lbl.pose.position = _pt(float(x), -float(z), 3.55)
    lbl.scale.z = 0.86
    lbl.color = C_WHITE
    lbl.text = "C"
    halo = _mk("ch_halo", mid, Marker.CYLINDER, fid, stamp)
    halo.pose.position = _pt(float(x), -float(z), 0.03)
    halo.scale = _sc(1.20, 1.20, 0.06)
    halo.color = _c(C_CHARGER.r, C_CHARGER.g, C_CHARGER.b, 0.26)
    return [cyl, lbl, halo]


def mk_station_base(x: int, z: int, mid: int, stamp, fid: str) -> Marker:
    m = _mk("st_pole", mid, Marker.CYLINDER, fid, stamp)
    m.pose.position = _pt(float(x), -float(z), 0.6)
    m.scale = _sc(0.35, 0.35, 1.2)
    m.color = _c(0.55, 0.55, 0.55)
    return m


# ─── dynamic markers ──────────────────────────────────────────────────────────
def mk_station_dyn(x: int, z: int, cid: int, is_target: bool,
                   mid: int, stamp, fid: str) -> List[Marker]:
    color = C_TARGET if is_target else C_STATION
    h = 2.7 if is_target else 1.4
    cap = _mk("st_cap", mid * 3, Marker.CYLINDER, fid, stamp, _DYN_LIFE_NS)
    cap.pose.position = _pt(float(x), -float(z), 1.2 + h / 2.0)
    cap.scale = _sc(0.56 if is_target else 0.52, 0.56 if is_target else 0.52, h)
    cap.color = color
    lbl = _mk("st_lbl", mid * 3 + 1, Marker.TEXT_VIEW_FACING, fid, stamp, _DYN_LIFE_NS)
    lbl.pose.position = _pt(float(x), -float(z), 1.2 + h + 0.5)
    lbl.scale.z = 0.66 if is_target else 0.60
    lbl.color = C_WHITE
    lbl.text = f"→{cid}" if is_target else str(cid)
    markers: List[Marker] = [cap, lbl]
    if is_target:
        ring = _mk("st_ring", mid * 3 + 2, Marker.CYLINDER, fid, stamp, _DYN_LIFE_NS)
        ring.pose.position = _pt(float(x), -float(z), 0.04)
        ring.scale = _sc(1.8, 1.8, 0.08)
        ring.color = _c(C_TARGET.r, C_TARGET.g, C_TARGET.b, 0.44)
        markers.append(ring)
        ring_outer = _mk("st_ring_outer", mid, Marker.CYLINDER, fid, stamp, _DYN_LIFE_NS)
        ring_outer.pose.position = _pt(float(x), -float(z), 0.03)
        ring_outer.scale = _sc(2.35, 2.35, 0.06)
        ring_outer.color = _c(C_TARGET.r, C_TARGET.g, C_TARGET.b, 0.16)
        markers.append(ring_outer)
    return markers


def mk_npc(nx: int, nz: int, mid: int, stamp, fid: str) -> List[Marker]:
    sp = _mk("npc_sp", mid * 2, Marker.SPHERE, fid, stamp, _DYN_LIFE_NS)
    sp.pose.position = _pt(float(nx), -float(nz), 2.2)
    sp.scale = _sc(0.70, 0.70, 0.70)
    sp.color = C_NPC
    disc = _mk("npc_disc", mid * 2 + 1, Marker.CYLINDER, fid, stamp, _DYN_LIFE_NS)
    disc.pose.position = _pt(float(nx), -float(nz), 0.05)
    disc.scale = _sc(6.0, 6.0, 0.08)
    disc.color = C_NPC_ZONE
    return [sp, disc]


def mk_drone_quad(hx: int, hz: int, dx: int, dz: int,
                  batt: int, batt_max: int, stamp, fid: str) -> List[Marker]:
    markers: List[Marker] = []
    rx, ry, z0 = float(hx), -float(hz), 2.2

    # Keep drone marker persistent so it remains visible even during sparse updates.
    body = _mk("d_body", 0, Marker.SPHERE, fid, stamp, 0)
    body.pose.position = _pt(rx, ry, z0)
    body.scale = _sc(0.52, 0.52, 0.28)
    body.color = C_DRONE
    markers.append(body)

    arms = _mk("d_arms", 1, Marker.LINE_LIST, fid, stamp, 0)
    arms.scale.x = 0.06
    arms.color = C_DRONE_ARM
    ar = 0.52
    for ax, ay in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        arms.points += [_pt(rx, ry, z0), _pt(rx + ax * ar, ry + ay * ar, z0)]
    markers.append(arms)

    for i, (ax, ay) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        rot = _mk("d_rotor", 2 + i, Marker.CYLINDER, fid, stamp, 0)
        rot.pose.position = _pt(rx + ax * ar, ry + ay * ar, z0 + 0.02)
        rot.scale = _sc(0.32, 0.32, 0.04)
        rot.color = _c(0.14, 0.14, 0.14, 0.90)
        markers.append(rot)

    if dx or dz:
        arr = _mk("d_arrow", 6, Marker.ARROW, fid, stamp, 0)
        arr.points = [_pt(rx, ry, z0), _pt(rx + dx * 0.70, ry - dz * 0.70, z0)]
        arr.scale = _sc(0.08, 0.18, 0.0)
        arr.color = C_WHITE
        markers.append(arr)

    ratio = float(batt) / max(batt_max, 1)
    bc = C_GREEN_BAR if ratio > 0.4 else (C_AMBER_BAR if ratio > 0.2 else C_RED_BAR)
    bar = _mk("d_batt", 7, Marker.LINE_STRIP, fid, stamp, 0)
    bar.scale.x = 0.18
    bar.color = bc
    bar.points = [_pt(rx - 0.40, ry, z0 + 0.50),
                  _pt(rx - 0.40 + 0.80 * ratio, ry, z0 + 0.50)]
    rail = _mk("d_rail", 8, Marker.LINE_STRIP, fid, stamp, 0)
    rail.scale.x = 0.08
    rail.color = _c(0.40, 0.40, 0.40)
    rail.points = [_pt(rx - 0.40, ry, z0 + 0.50), _pt(rx + 0.40, ry, z0 + 0.50)]
    markers += [bar, rail]
    return markers


def mk_trail(positions: list, stamp, fid: str) -> Optional[Marker]:
    if len(positions) < 2:
        return None
    m = _mk("trail", 0, Marker.LINE_STRIP, fid, stamp, 0)
    m.scale.x = 0.12
    m.color = C_TRAIL
    for hx, hz in positions:
        m.points.append(_pt(float(hx), -float(hz), 2.2))
    return m


def mk_status(snap: dict, stamp, fid: str) -> Marker:
    hx = int(snap["hero"]["pos"][0])
    hz = int(snap["hero"]["pos"][1])
    batt = int(snap["hero"]["battery"])
    batt_max = int(snap["hero"]["battery_max"])
    pct = int(batt * 100 / max(batt_max, 1))
    t = _mk("status", 0, Marker.TEXT_VIEW_FACING, fid, stamp, 0)
    t.pose.position = _pt(float(hx), -float(hz), 3.2)
    t.scale.z = 0.60
    t.color = C_YELLOW
    t.text = f"S{snap['step_no']}/{snap['max_step']}  {pct}%  x{snap['hero']['delivered']}"
    return t


# ─── node ─────────────────────────────────────────────────────────────────────
class VizNode(Node):
    _TRAIL_LEN = 120

    def __init__(self) -> None:
        super().__init__("viz_node")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("record_gif", False)
        self.declare_parameter("gif_out_dir", "ros2_ws/gif_out")
        self.declare_parameter("gif_fps", 6)
        self.declare_parameter("show_all_elements", True)
        self.declare_parameter("render_ack_topic", "/drone/render_ack")
        self.declare_parameter("episode_sync_wait_sec", 0.05)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.record_gif = bool(self.get_parameter("record_gif").value)
        self.gif_fps = int(self.get_parameter("gif_fps").value)
        self.show_all_elements = bool(self.get_parameter("show_all_elements").value)
        self.render_ack_topic = str(self.get_parameter("render_ack_topic").value)
        self.episode_sync_wait_sec = float(self.get_parameter("episode_sync_wait_sec").value)
        gif_dir = str(self.get_parameter("gif_out_dir").value)

        from drone_grid_sim._bootstrap import DEFAULT_REPO_ROOT
        repo = os.environ.get("KAIWU_REPO", DEFAULT_REPO_ROOT)
        if not os.path.isabs(gif_dir):
            gif_dir = os.path.join(repo, gif_dir)
        self.gif_dir = gif_dir
        os.makedirs(self.gif_dir, exist_ok=True)

        self.recorder = None
        if self.record_gif:
            try:
                from closeloop.render import FrameRecorder
                self.recorder = FrameRecorder(self.gif_dir, episode_id=0, fps=self.gif_fps)
                self.get_logger().info(f"[viz] GIF recorder -> {self.gif_dir}")
            except Exception as exc:
                self.get_logger().warning(f"[viz] GIF disabled: {exc}")

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        cloud_stream_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        snap_qos = QoSProfile(
            depth=200,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # RACER-style PointCloud2 publishers
        self.pub_occ_cloud = self.create_publisher(
            PointCloud2, "/drone/occupancy_cloud", cloud_stream_qos)
        self.pub_unknown_cloud = self.create_publisher(
            PointCloud2, "/drone/unknown_cloud", cloud_stream_qos)
        # MarkerArray publishers (existing)
        self.pub_static  = self.create_publisher(MarkerArray, "/drone/markers_static",  latched)
        # Keep only latest dynamic frame to avoid marker lag behind point clouds.
        self.pub_dynamic = self.create_publisher(MarkerArray, "/drone/markers_dynamic", 1)
        self.pub_render_ack = self.create_publisher(Int32, self.render_ack_topic, 10)

        # Individual marker publishers for trajectory group
        self.pub_drone_marker = self.create_publisher(Marker, "/drone/drone_marker", 1)
        self.pub_trail_marker = self.create_publisher(Marker, "/drone/trail_marker", 1)

        self.create_subscription(String, "/drone/snapshot",      self._on_snapshot, snap_qos)
        self.create_subscription(String, "/drone/episode_event", self._on_event,    10)

        self._episode = -1
        self._static_episode = -1
        self._static_map_sig: Optional[Tuple[int, int, int, int]] = None
        self._need_static_refresh = True
        self._point_cloud_sig: Optional[Tuple[Any, ...]] = None
        self._point_cloud_cache: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._cur_reward = 0.0
        self._frames = 0
        self._trail: collections.deque = collections.deque(maxlen=self._TRAIL_LEN)
        self._last_step_no = -1
        self._last_episode = -1
        self._stable_explored_mask: Optional[np.ndarray] = None
        self._pending_episode_snapshot: Optional[dict] = None
        self._pending_snapshot_monotonic = 0.0

    def _enter_episode(self, ep: int, reset_recorder: bool = True) -> None:
        """Apply episode boundary atomically for both event-first and snapshot-first order."""
        self._episode = ep
        self._last_episode = ep
        self._frames = 0
        self._trail.clear()
        self._need_static_refresh = True
        self._last_step_no = -1
        self._stable_explored_mask = None
        if reset_recorder and self.recorder is not None:
            self.recorder.reset(episode_id=ep)

    def _on_snapshot(self, msg: String) -> None:
        try:
            snap = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        now_mono = time.monotonic()

        # During episode boundary, wait briefly for reset event to avoid
        # rendering cloud from new episode together with old dynamic markers.
        snap_episode = int(snap.get("episode", self._episode))
        if snap_episode > self._episode:
            pending_ep = -1
            if self._pending_episode_snapshot is not None:
                pending_ep = int(self._pending_episode_snapshot.get("episode", -1))

            if pending_ep != snap_episode:
                self._pending_episode_snapshot = snap
                self._pending_snapshot_monotonic = now_mono
                return

            self._pending_episode_snapshot = snap
            if (now_mono - self._pending_snapshot_monotonic) < max(0.0, self.episode_sync_wait_sec):
                return

            # Fallback: if reset event is delayed/lost, trust snapshot episode
            # to avoid sync deadlock.
            self.get_logger().warning(
                f"[viz] reset event wait timeout, fallback to snapshot episode={snap_episode}"
            )
            self._enter_episode(snap_episode, reset_recorder=True)
            self._pending_episode_snapshot = None

        self._process_snapshot(snap)

    def _process_snapshot(self, snap: dict) -> None:
        step_no = int(snap.get("step_no", snap.get("frame_no", 0)))
        snap_episode = int(snap.get("episode", self._episode))

        # Snapshot can arrive before reset event; enter new episode immediately.
        if self._episode >= 0 and snap_episode < self._episode:
            return
        if snap_episode > self._episode:
            self._enter_episode(snap_episode, reset_recorder=True)

        # Drop stale frames within an episode to prevent render rollback flicker.
        if self._last_step_no >= 0 and step_no < self._last_step_no:
            return
        self._last_step_no = step_no

        explored = snap.get("explored_mask")
        if isinstance(explored, list):
            cur_mask = np.asarray(explored, dtype=np.int8)
            if self._stable_explored_mask is None or self._stable_explored_mask.shape != cur_mask.shape:
                self._stable_explored_mask = cur_mask.copy()
            else:
                # Keep explored area monotonic and physically plausible:
                # newly explored cells must lie within current sensor range.
                hero = snap.get("hero", {})
                hpos = hero.get("pos", (0, 0)) if isinstance(hero, dict) else (0, 0)
                hx, hz = int(hpos[0]), int(hpos[1])
                sensor_r = int(snap.get("sensor_range_cells", 10))

                newly = (cur_mask == 1) & (self._stable_explored_mask == 0)
                if np.any(newly):
                    zz, xx = np.nonzero(newly)
                    keep = np.maximum(np.abs(xx - hx), np.abs(zz - hz)) <= sensor_r
                    if np.any(keep):
                        self._stable_explored_mask[zz[keep], xx[keep]] = 1
            snap["explored_mask"] = self._stable_explored_mask.tolist()

        # Rebuild static layer only when a new episode starts. Frequent
        # static DELETEALL+recreate cycles can cause visible flicker in RViz.
        if self._need_static_refresh:
            map_sig = self._map_signature(snap)
            self._publish_static(snap)
            self._static_map_sig = map_sig
            self._need_static_refresh = False
        if self._episode != self._static_episode:
            self._static_episode = self._episode
            self._trail.clear()

        # Point cloud updates every frame — explored_mask changes each step
        self._publish_pointcloud(snap)

        hpos = snap["hero"]["pos"]
        self._trail.append((int(hpos[0]), int(hpos[1])))
        self._publish_dynamic(snap)

        if self.recorder is not None:
            self._record_frame(snap)

        # Step-sync handshake: ACK this frame after rendering callbacks finish.
        self.pub_render_ack.publish(Int32(data=int(snap.get("step_no", 0))))

    def _on_event(self, msg: String) -> None:
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        kind = ev.get("kind")
        ep = int(ev.get("episode", 0))
        self._cur_reward = float(ev.get("total_env_reward", 0.0))
        if kind == "reset":
            # Ignore stale reset events from older episodes.
            if ep < self._episode:
                return
            # If snapshot already switched episode, this becomes idempotent.
            if ep > self._episode:
                self._enter_episode(ep, reset_recorder=True)
            if self._pending_episode_snapshot is not None:
                pending_ep = int(self._pending_episode_snapshot.get("episode", -1))
                if pending_ep == ep:
                    snap = self._pending_episode_snapshot
                    self._pending_episode_snapshot = None
                    self._process_snapshot(snap)
        elif kind == "done":
            if self.recorder is not None and self._frames > 0:
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = self.recorder.save_gif(f"episode_{ep:04d}_{ts}.gif")
                self.get_logger().info(f"[viz] GIF saved: {path}  ({self._frames} frames)")

    def _map_signature(self, snap: dict) -> Tuple[int, int, int, int]:
        grid = snap["grid"]
        height = len(grid)
        width = len(grid[0]) if height else 0
        obstacle_count = sum(sum(int(cell) for cell in row) for row in grid)
        # Static layer must depend on map geometry only. If it depends on
        # exploration progress, RViz receives frequent DELETEALL/rebuild bursts
        # and appears to flicker.
        return (height, width, obstacle_count, 0)

    def _publish_pointcloud(self, snap: dict) -> None:
        """Publish RACER-style PointCloud2: explored obstacles + unexplored fog."""
        grid = snap["grid"]
        height_map = snap.get("height_map")
        explored = snap.get("explored_mask")
        if explored is None:
            H = len(grid)
            W = len(grid[0]) if H > 0 else 0
            explored = [[1] * W for _ in range(H)]
        now = self.get_clock().now().to_msg()

        explored_np = np.asarray(explored, dtype=np.int8)
        all_points = self._get_map_point_cloud(snap)
        visible_points = all_points if self.show_all_elements else filter_point_cloud_by_explored(all_points, explored_np)

        occ_msg = _build_explored_obstacle_cloud(
            visible_points, self.frame_id, now,
        )
        self.pub_occ_cloud.publish(occ_msg)

        # Unexplored fog cloud
        if not self.show_all_elements:
            unk_msg = _build_unexplored_cloud(grid, explored, self.frame_id, now)
            self.pub_unknown_cloud.publish(unk_msg)

    def _get_map_point_cloud(self, snap: dict) -> np.ndarray:
        grid = np.asarray(snap["grid"], dtype=np.int8)
        height_map = snap.get("height_map")
        if height_map is None:
            height_map_np = np.zeros_like(grid, dtype=np.float32)
        else:
            height_map_np = np.asarray(height_map, dtype=np.float32)
        map_source = str(snap.get("map_source", "city"))
        map_name = str(snap.get("racer_map_name", "pillar"))
        racer_resolution = float(snap.get("racer_resolution", 0.5))
        racer_height_thresh = float(snap.get("racer_height_thresh", 0.15))
        sig = (
            map_source,
            map_name,
            racer_resolution,
            racer_height_thresh,
            grid.shape,
            int(np.sum(grid)),
        )
        if sig == self._point_cloud_sig:
            return self._point_cloud_cache

        # Always voxelize from grid+height map to get visually solid obstacles.
        # This also keeps official_json and racer_pcd rendering consistent.
        point_cloud = height_map_to_point_cloud(grid, height_map_np)

        self._point_cloud_sig = sig
        self._point_cloud_cache = point_cloud
        return self._point_cloud_cache

    def _publish_static(self, snap: dict) -> None:
        """Publish entity markers only — no ground/road/building markers.

        Map geometry is rendered entirely via RACER-style point clouds.
        """
        now = self.get_clock().now().to_msg()
        explored = snap.get("explored_mask")

        arr = MarkerArray()
        clr = Marker()
        clr.action = Marker.DELETEALL
        clr.header.frame_id = self.frame_id
        clr.header.stamp = now
        arr.markers.append(clr)

        explored = snap.get("explored_mask")

        for i, pos in enumerate(snap["warehouses"]):
            markers = mk_warehouse(int(pos[0]), int(pos[1]), i, now, self.frame_id)
            arr.markers += markers
        for i, pos in enumerate(snap["chargers"]):
            markers = mk_charger(int(pos[0]), int(pos[1]), i, now, self.frame_id)
            arr.markers += markers
        for i, (pos, _cid) in enumerate(snap["stations"]):
            marker = mk_station_base(int(pos[0]), int(pos[1]), i, now, self.frame_id)
            arr.markers.append(marker)

        self.get_logger().info(
            f"[viz] entity markers published: {len(arr.markers)} markers"
        )
        self.pub_static.publish(arr)

    def _publish_dynamic(self, snap: dict) -> None:
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        target_ids = set(int(c) for c in snap.get("target_ids", []))
        explored = snap.get("explored_mask")

        for i, (pos, cid) in enumerate(snap["stations"]):
            station_markers = mk_station_dyn(
                int(pos[0]), int(pos[1]), int(cid),
                int(cid) in target_ids, i, now, self.frame_id,
            )
            if (not self.show_all_elements) and (not _cell_is_explored(explored, tuple(pos))):
                station_markers = _gray_markers(station_markers)
            arr.markers += station_markers
        for i, pos in enumerate(snap["npcs"]):
            if self.show_all_elements or _cell_is_explored(explored, tuple(pos)):
                arr.markers += mk_npc(int(pos[0]), int(pos[1]), i, now, self.frame_id)

        trail_m = mk_trail(list(self._trail), now, self.frame_id)
        if trail_m is not None:
            arr.markers.append(trail_m)

        hpos = snap["hero"]["pos"]
        hx, hz = int(hpos[0]), int(hpos[1])
        dx, dz = tuple(snap.get("last_action_delta", (0, 0)))
        drone_markers = mk_drone_quad(
            hx, hz, int(dx), int(dz),
            int(snap["hero"]["battery"]),
            int(snap["hero"]["battery_max"]),
            now, self.frame_id,
        )
        arr.markers += drone_markers
        arr.markers.append(mk_status(snap, now, self.frame_id))
        self.pub_dynamic.publish(arr)

        # Publish individual drone marker for trajectory group
        if drone_markers:
            self.pub_drone_marker.publish(drone_markers[0])
        if trail_m is not None:
            self.pub_trail_marker.publish(trail_m)

    def _record_frame(self, snap: dict) -> None:
        try:
            import numpy as np
            s = dict(snap)
            s["grid"] = np.array(snap["grid"], dtype=np.int8)
            if "cell_types" in snap and snap["cell_types"] is not None:
                s["cell_types"] = np.array(snap["cell_types"], dtype=np.int16)
            if "height_map" in snap and snap["height_map"] is not None:
                s["height_map"] = np.array(snap["height_map"], dtype=np.float32)
            if "explored_mask" in snap and snap["explored_mask"] is not None:
                s["explored_mask"] = np.array(snap["explored_mask"], dtype=np.int8)
            s["hero"] = {**snap["hero"], "pos": tuple(snap["hero"]["pos"])}
            s["warehouses"] = [tuple(p) for p in snap["warehouses"]]
            s["chargers"]   = [tuple(p) for p in snap["chargers"]]
            s["stations"]   = [(tuple(p), int(c)) for p, c in snap["stations"]]
            s["npcs"]       = [tuple(p) for p in snap["npcs"]]
            s["last_action_delta"] = tuple(snap.get("last_action_delta", (0, 0)))
            self.recorder.render(s, reward_so_far=self._cur_reward)
            self._frames += 1
        except Exception as exc:
            self.get_logger().warning(
                "[viz] frame error: %r | types grid=%s cell_types=%s explored=%s" % (
                    exc,
                    type(snap.get("grid")).__name__,
                    type(snap.get("cell_types")).__name__,
                    type(snap.get("explored_mask")).__name__,
                )
            )


def main() -> None:
    rclpy.init()
    node = VizNode()
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
