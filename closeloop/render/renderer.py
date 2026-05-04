"""Matplotlib renderer that turns ``env.snapshot()`` frames into PNG/GIF."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib

from closeloop.env.racer_map import (
    filter_point_cloud_by_explored,
    height_map_to_point_cloud,
    load_racer_grid_frame_point_cloud,
)
from closeloop.env.official_map import load_cell_type_palette

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Circle  # noqa: E402


# RACER-style colour palette — light cream background matching RACER's rviz config.
COLOR_FREE = "#fffff3"
COLOR_BUILDING = "#a4a4a4"
COLOR_FOG = "#2a2a2e"
COLOR_WAREHOUSE = "#1976d2"
COLOR_CHARGER = "#fbc02d"
COLOR_STATION = "#43a047"
COLOR_TARGET = "#e53935"
COLOR_NPC = "#8e24aa"
COLOR_HERO = "#ff7043"
COLOR_UNKNOWN = "#808080"

# Action arrow labels (must match closeloop.env.grid_env.ACTION_DELTA)
ACTION_NAMES = ["R", "RU", "U", "LU", "L", "LD", "D", "RD"]


class FrameRecorder:
    """Render env snapshots to PNG; optionally export GIF when episode ends."""

    def __init__(
        self,
        out_dir: str,
        episode_id: int = 0,
        cell_px: int = 12,
        max_frames: int = 4000,
        fps: int = 8,
    ):
        self.out_dir = out_dir
        self.episode_id = episode_id
        self.cell_px = cell_px
        self.max_frames = max_frames
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self._point_cloud_sig: Optional[tuple] = None
        self._point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._cell_palette = load_cell_type_palette()
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def render(self, snap: Dict[str, Any], reward_so_far: float = 0.0) -> np.ndarray:
        """Render one snapshot and append it to the buffer.

        Returns the RGB ndarray for the rendered frame.
        """

        if len(self.frames) >= self.max_frames:
            return self.frames[-1]

        grid: np.ndarray = snap["grid"]
        H, W = grid.shape
        explored = snap.get("explored_mask")
        if explored is not None:
            explored = np.array(explored, dtype=np.int8)
        else:
            explored = np.ones((H, W), dtype=np.int8)

        fig_w = max(4.0, W * self.cell_px / 100.0)
        fig_h = max(4.0, H * self.cell_px / 100.0) + 0.6  # extra for title strip
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        canvas = self._build_map_canvas(snap, explored)
        ax.imshow(canvas, origin="upper", interpolation="nearest")

        visible_cloud = self._get_point_cloud(snap)
        if visible_cloud.size > 0:
            ax.scatter(
                visible_cloud[:, 0],
                visible_cloud[:, 1],
                c=np.clip(visible_cloud[:, 2], 0.0, None),
                cmap="turbo",
                s=max(4.0, self.cell_px * 0.30),
                alpha=0.85,
                linewidths=0.0,
                zorder=2,
            )

        # entities
        for (x, z) in snap["warehouses"]:
            _square(ax, x, z, COLOR_WAREHOUSE, label="W")
        for (x, z) in snap["chargers"]:
            _square(ax, x, z, COLOR_CHARGER, label="C")
        targets = set(snap["target_ids"])
        for (pos, cid) in snap["stations"]:
            color = COLOR_TARGET if cid in targets else COLOR_STATION
            _square(ax, pos[0], pos[1], color, label="T" if cid in targets else "S")
        for (x, z) in snap["npcs"]:
            ax.add_patch(Circle((x, z), 0.42, color=COLOR_NPC, zorder=4))

        hx, hz = snap["hero"]["pos"]
        ax.add_patch(Circle((hx, hz), 0.45, color=COLOR_HERO, zorder=5))
        # last-action arrow (drawn from previous virtual position)
        dx, dz = snap.get("last_action_delta", (0, 0))
        if dx or dz:
            ax.annotate(
                "",
                xy=(hx + dx, hz + dz),
                xytext=(hx, hz),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                zorder=6,
            )

        # collision flash
        ci = snap.get("collision_npc_idx")
        if ci is not None and ci < len(snap["npcs"]):
            cx, cz = snap["npcs"][ci]
            ax.add_patch(Circle((cx, cz), 0.85, fill=False, ec="red", lw=2, zorder=7))

        title = (
            f"ep{self.episode_id} step={snap['step_no']}/{snap['max_step']} "
            f"batt={snap['hero']['battery']}/{snap['hero']['battery_max']} "
            f"deliv={snap['hero']['delivered']} "
            f"act={ACTION_NAMES[snap['last_action']] if snap['last_action'] >= 0 else '-'} "
            f"R={reward_so_far:.2f}"
        )
        if snap["terminated"] or snap["truncated"]:
            title += f"  [{snap['result_message'] or ('truncated' if snap['truncated'] else 'done')}]"
        ax.set_title(title, fontsize=9)

        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)  # invert y so z+ is "down" / "south"
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        # Help debug coordinate assumptions: origin is always map corner, not hero spawn.
        ax.text(0.0, 0.0, "(0,0)", color="black", fontsize=7, ha="left", va="bottom", zorder=8)
        ax.text(W - 1.0, H - 1.0, f"({W-1},{H-1})", color="black", fontsize=7, ha="right", va="top", zorder=8)
        fig.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        rgba = np.asarray(buf)
        rgb = rgba[..., :3].copy()
        plt.close(fig)
        self.frames.append(rgb)
        return rgb

    def _visible_obstacle_points(self, snap: Dict[str, Any], explored: np.ndarray) -> np.ndarray:
        point_cloud = self._get_point_cloud(snap)
        if point_cloud.size == 0:
            return point_cloud
        return filter_point_cloud_by_explored(point_cloud, explored)

    def _get_point_cloud(self, snap: Dict[str, Any]) -> np.ndarray:
        grid: np.ndarray = snap["grid"]
        height_map: Optional[np.ndarray] = snap.get("height_map")
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
            return self._point_cloud

        if map_source == "racer_pcd":
            point_cloud, _origin = load_racer_grid_frame_point_cloud(
                map_name=map_name,
                resolution=racer_resolution,
                obstacle_height_thresh=racer_height_thresh,
            )
        else:
            if height_map is None:
                height_map = np.zeros_like(grid, dtype=np.float32)
            point_cloud = height_map_to_point_cloud(grid, np.asarray(height_map, dtype=np.float32))

        self._point_cloud_sig = sig
        self._point_cloud = point_cloud
        return self._point_cloud

    @staticmethod
    def _is_explored(explored: np.ndarray, x: int, z: int) -> bool:
        h, w = explored.shape
        return 0 <= x < w and 0 <= z < h and bool(explored[z, x])

    def _build_map_canvas(self, snap: Dict[str, Any], explored: np.ndarray) -> np.ndarray:
        grid = np.asarray(snap["grid"], dtype=np.int8)
        cell_types = snap.get("cell_types")
        if cell_types is not None:
            cell_types = np.asarray(cell_types, dtype=np.int16)

        h, w = grid.shape
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        for z in range(h):
            for x in range(w):
                canvas[z, x] = self._cell_color(cell_types, grid, x, z)
        return canvas

    def _cell_color(
        self,
        cell_types: Optional[np.ndarray],
        grid: np.ndarray,
        x: int,
        z: int,
    ) -> np.ndarray:
        if cell_types is not None:
            rgba = self._cell_palette.get(int(cell_types[z, x]))
            if rgba is not None:
                return np.array(rgba[:3], dtype=np.float32)
        if grid[z, x]:
            return np.array(matplotlib.colors.to_rgb(COLOR_BUILDING), dtype=np.float32)
        return np.array(matplotlib.colors.to_rgb(COLOR_FREE), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def save_gif(self, filename: Optional[str] = None) -> Optional[str]:
        if not self.frames:
            return None
        if filename is None:
            filename = f"episode_{self.episode_id:04d}.gif"
        path = os.path.join(self.out_dir, filename)

        frames = self._normalize_frames(self.frames)

        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            try:
                import imageio  # type: ignore
            except ImportError:
                imageio = None  # type: ignore

        if imageio is not None:
            imageio.mimsave(path, frames, duration=1.0 / max(self.fps, 1))
            return path

        # Fallback: write PNGs and call ImageMagick `convert`.
        png_dir = os.path.join(self.out_dir, f"_frames_ep{self.episode_id:04d}")
        os.makedirs(png_dir, exist_ok=True)
        from PIL import Image  # pillow ships with matplotlib

        png_paths = []
        for i, frame in enumerate(frames):
            p = os.path.join(png_dir, f"f{i:05d}.png")
            Image.fromarray(frame).save(p)
            png_paths.append(p)

        import shutil
        import subprocess

        convert_bin = shutil.which("convert") or shutil.which("magick")
        if convert_bin is None:
            return png_dir  # at least we have PNGs
        delay = max(1, int(round(100.0 / max(self.fps, 1))))
        cmd = [convert_bin, "-delay", str(delay), "-loop", "0", *png_paths, path]
        subprocess.run(cmd, check=False)
        return path

    def save_last_png(self, filename: Optional[str] = None) -> Optional[str]:
        if not self.frames:
            return None
        from PIL import Image

        if filename is None:
            filename = f"episode_{self.episode_id:04d}_last.png"
        path = os.path.join(self.out_dir, filename)
        Image.fromarray(self.frames[-1]).save(path)
        return path

    def reset(self, episode_id: Optional[int] = None) -> None:
        self.frames.clear()
        if episode_id is not None:
            self.episode_id = episode_id

    @staticmethod
    def _normalize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        max_h = max(frame.shape[0] for frame in frames)
        max_w = max(frame.shape[1] for frame in frames)
        normalized: List[np.ndarray] = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h == max_h and w == max_w:
                normalized.append(frame)
                continue
            canvas = np.full((max_h, max_w, frame.shape[2]), 255, dtype=frame.dtype)
            canvas[:h, :w] = frame
            normalized.append(canvas)
        return normalized


# ---------------------------------------------------------------------------- #
def _square(ax, x: int, z: int, color: str, label: str = "") -> None:
    ax.add_patch(
        Rectangle((x - 0.5, z - 0.5), 1, 1, facecolor=color, edgecolor="black", lw=0.4, zorder=3)
    )
    if label:
        ax.text(x, z, label, ha="center", va="center", fontsize=6, color="white", zorder=4)


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    lum = float(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    gray = 0.35 + 0.55 * lum
    return np.array([gray, gray, gray], dtype=np.float32)


def _gray_hex(color: str) -> str:
    rgb = np.array(matplotlib.colors.to_rgb(color), dtype=np.float32)
    g = _to_grayscale(rgb)
    return matplotlib.colors.to_hex((float(g[0]), float(g[1]), float(g[2])))
