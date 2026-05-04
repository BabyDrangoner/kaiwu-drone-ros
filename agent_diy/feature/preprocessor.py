#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY Preprocessor for Drone Delivery.

负责:
1. 把 env_obs 解析成定长特征向量 (配置鲁棒: K slot + padding).
2. 维护跨步状态(历史动作、历史位置、已访问格子集合) 以支持 novelty / stuck 检测.
3. 计算奖励(reward shaping), 主要应对: 耗电, NPC, 卡角落.
"""

from collections import deque

import numpy as np

from agent_diy.conf.conf import Config


# 地图坐标范围: 每张图最大 128, 所以绝对坐标用 /128 归一化
MAP_COORD_SCALE = 128.0
MAX_DIST = 1.41 * MAP_COORD_SCALE  # 对角线距离上界


def _norm(v, lo, hi):
    """Linear normalize to [0, 1]."""
    v = np.clip(v, lo, hi)
    return (v - lo) / max(hi - lo, 1e-6)


def _sym_norm(v, scale):
    """Symmetric normalize, result in [-1, 1]."""
    return float(np.clip(v / max(scale, 1e-6), -1.0, 1.0))


# 8 个动作对应的位移向量 (dx, dz), 用于预测"下一步理论位置" 以检测 bounce / stuck
ACTION_DELTA = np.array(
    [
        [1, 0],    # 0 右
        [1, -1],   # 1 右上
        [0, -1],   # 2 上
        [-1, -1],  # 3 左上
        [-1, 0],   # 4 左
        [-1, 1],   # 5 左下
        [0, 1],    # 6 下
        [1, 1],    # 7 右下
    ],
    dtype=np.int32,
)


class Preprocessor:
    """Feature + reward preprocessor."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    # 生命周期                                                            #
    # ------------------------------------------------------------------ #
    def reset(self):
        # 即时状态
        self.cur_pos = (0, 0)
        self.prev_pos = (0, 0)
        self.battery = 100
        self.battery_max = 300
        self.max_step = 1000
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0
        self.charge_count = 0
        self.last_charge_count = 0
        self.back_warehouse_count = 0
        self.last_back_warehouse_count = 0

        # 实体
        self.stations = []   # list of organ dict, sub_type=3
        self.chargers = []   # sub_type=2
        self.warehouses = [] # sub_type=1
        self.npcs = []

        # 环境 meta (env_info)
        self.npc_count_meta = 4       # 本局 NPC 数量, 根据 usr_conf 估算
        self.station_count_meta = 10
        self.charger_count_meta = 4

        # 局部视野 map_info (21x21)
        self.map_info = None

        # 历史
        self.action_history = deque([-1] * Config.ACTION_HISTORY_LEN, maxlen=Config.ACTION_HISTORY_LEN)
        self.pos_history = deque(maxlen=16)     # 最近 16 步位置 (stuck 检测)
        self.visited_set = set()                # 已到访格子 (novelty)
        self.last_action = -1

        # 奖励计算辅助
        self.prev_target_dist = None
        self.prev_charger_dist = None          # 【续训 v2】低电势函数用
        self.prev_npc_dist = None              # 【续训 v2】evade_move 判定用
        self.last_delivery_step = -9999        # 【续训 v2】combo 判定用
        self.prev_battery = None
        self.bounce_flag = 0
        self.stuck_flag = 0
        self.legal_act = [1] * 8

        # NPC approach 追踪: 上一帧按"距自身距离升序"排列的 NPC 位置列表,
        # 当前帧用最近邻匹配回找 -> approach = prev_dist - cur_dist.
        # 不用 id 是因为协议里 NPC 无稳定 id, 用空间最近邻足够稳健(NPC 最多 4 个,
        # 一步位移 <=2 格, 帧间最近邻基本不会错配).
        self.prev_npc_positions = []  # list[(x, z)], 上一帧看到的 NPC 绝对坐标
        # 【action_aux 独立缓存】因为 _build_npc_feat 会在自己末尾更新
        # prev_npc_positions 为"当前帧", 若 action_aux 也用它就失去跨帧差异.
        # 所以这里单独维护一份, 只在 _build_action_aux_feat 里读 + 更新.
        self.prev_npc_positions_for_aux = []
        # 供奖励使用: 当前帧最近 NPC 的 approach_rate (>0 表示正在逼近)
        self.cur_npc_approach_rate = 0.0
        self.cur_npc_threat = 0.0

        # ---- episode-level 诊断计数 ---- #
        # workflow 会在局末读这些值上报, 用于监控面板诊断
        self.ep_bounce_count = 0       # 本局撞墙次数
        self.ep_stuck_count = 0        # 本局被判 stuck 的步数
        self.ep_min_npc_dist = None    # 本局出现过的最小 NPC 距离
        self.ep_low_batt_steps = 0     # 本局电量 <20% 的步数
        self.last_npc_dist = None      # 最近一步的最小 NPC 距离(供局末原因判定)

    # ------------------------------------------------------------------ #
    # 解析 env_obs                                                        #
    # ------------------------------------------------------------------ #
    def _parse(self, env_obs):
        obs = env_obs["observation"]
        frame = obs["frame_state"]
        hero = frame["heroes"]

        self.prev_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))
        self.prev_battery = self.battery
        self.battery = int(hero.get("battery", self.battery))
        self.battery_max = int(hero.get("battery_max", self.battery_max))
        self.packages = list(hero.get("packages", []))
        self.last_delivered = self.delivered
        self.delivered = int(hero.get("delivered", 0))
        self.step_no = int(obs.get("step_no", 0))

        env_info = obs.get("env_info", {})
        self.last_charge_count = self.charge_count
        self.charge_count = int(
            env_info.get("charge_count", env_info.get("charger_count", self.charge_count))
        )
        self.last_back_warehouse_count = self.back_warehouse_count
        self.back_warehouse_count = int(
            env_info.get(
                "back_warehouse_count",
                env_info.get("warehouse_count", self.back_warehouse_count),
            )
        )
        self.max_step = int(env_info.get("max_step", self.max_step))
        self.charger_count_meta = int(
            env_info.get(
                "total_charger",
                env_info.get("charger_station_count", self.charger_count_meta),
            )
        )
        self.station_count_meta = int(
            env_info.get(
                "total_post_station",
                env_info.get("station_count", self.station_count_meta),
            )
        )

        self.stations, self.chargers, self.warehouses = [], [], []
        for organ in frame.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 1:
                self.warehouses.append(organ)
            elif st == 2:
                self.chargers.append(organ)
            elif st == 3:
                self.stations.append(organ)

        npcs_raw = list(frame.get("npcs", []))
        self.npcs = []
        for npc in npcs_raw:
            if int(npc.get("is_in_view", 1)) == 0:
                continue
            pos = npc.get("pos", {}) if isinstance(npc, dict) else {}
            nx = int(pos.get("x", -1))
            nz = int(pos.get("z", -1))
            if nx < 0 or nz < 0:
                continue
            self.npcs.append(npc)
        # NPC 数量可推断
        if self.npcs:
            self.npc_count_meta = max(self.npc_count_meta, len(self.npcs))

        # map_info 在协议里写的是 MapInfo.map_info = list[list[int]], 但实际环境
        # 可能直接给 list[list[int]], 也可能给 {"map_info": list[list[int]]},
        # 两种都兼容一下避免挂掉
        mi = obs.get("map_info", None)
        if isinstance(mi, dict):
            mi = mi.get("map_info", None)
        self.map_info = mi

        # 合法动作字段名也双写兜底(协议写 legal_act, 但有些版本是 legal_action)
        legal = obs.get("legal_act", None)
        if legal is None:
            legal = obs.get("legal_action", None)
        self.legal_act = legal if legal else [1] * 8

    # ------------------------------------------------------------------ #
    # 特征构造                                                            #
    # ------------------------------------------------------------------ #
    def feature_process(self, env_obs, last_action):
        self._parse(env_obs)
        self.last_action = last_action

        # 维护历史
        if last_action >= 0:
            self.action_history.append(int(last_action))
        self.pos_history.append(self.cur_pos)
        is_new = self.cur_pos not in self.visited_set
        self.visited_set.add(self.cur_pos)

        # 1) self state
        self_feat = self._build_self_feat()
        # 2) action history one-hot
        act_hist_feat = self._build_action_hist_feat()
        # 3) stuck features
        stuck_feat = self._build_stuck_feat()
        # 4) env meta
        env_meta_feat = self._build_env_meta_feat()

        # 5) entities (定长 slot)
        station_feat = self._build_station_feat()
        npc_feat = self._build_npc_feat()
        charger_feat = self._build_charger_feat()
        warehouse_feat = self._build_warehouse_feat()

        # 6) local map (21x21 -> 7x7 via max-pool-3)
        map_feat = self._build_local_map_feat()

        # 7) 【action_aux 阶段新增】动作级辅助信号 (27 维) — 专解决避碰问题
        action_aux_feat = self._build_action_aux_feat()

        feature = np.concatenate(
            [
                self_feat,
                act_hist_feat,
                stuck_feat,
                env_meta_feat,
                station_feat,
                npc_feat,
                charger_feat,
                warehouse_feat,
                map_feat,
                action_aux_feat,
            ]
        ).astype(np.float32)

        assert feature.shape[0] == Config.FEATURE_LEN, (
            f"feature dim mismatch: got {feature.shape[0]}, expect {Config.FEATURE_LEN}"
        )

        legal_action = self._build_legal_action()
        reward = self._compute_reward(is_new_pos=is_new)

        return feature, legal_action, reward

    # ------------------------------------------------------------------ #
    # 子模块: self / history / stuck / env_meta                           #
    # ------------------------------------------------------------------ #
    def _build_self_feat(self):
        batt_ratio = _norm(self.battery, 0, self.battery_max)
        batt_low = 1.0 if batt_ratio < 0.25 else 0.0
        pkg_ratio = _norm(len(self.packages), 0, 3)
        has_pkg = 1.0 if len(self.packages) > 0 else 0.0
        step_ratio = _norm(self.step_no, 0, self.max_step)
        pos_x = _sym_norm(self.cur_pos[0], MAP_COORD_SCALE)
        pos_y = _sym_norm(self.cur_pos[1], MAP_COORD_SCALE)
        delivered_norm = _norm(self.delivered, 0, 30)   # 30 作为大概上界
        charge_norm = _norm(self.charge_count, 0, 20)
        back_warehouse_norm = _norm(self.back_warehouse_count, 0, 10)
        return np.array(
            [
                batt_ratio,
                batt_low,
                pkg_ratio,
                has_pkg,
                step_ratio,
                pos_x,
                pos_y,
                delivered_norm,
                charge_norm,
                back_warehouse_norm,
            ],
            dtype=np.float32,
        )

    def _build_action_hist_feat(self):
        hist = np.zeros(Config.ACTION_HIST_DIM, dtype=np.float32)
        for i, a in enumerate(self.action_history):
            if 0 <= a < Config.ACTION_NUM:
                hist[i * Config.ACTION_NUM + a] = 1.0
        return hist

    def _build_stuck_feat(self):
        # 位置方差(最近 16 步归一化 stdev) — 卡角落时 0
        if len(self.pos_history) >= 4:
            arr = np.array(self.pos_history, dtype=np.float32)
            var = arr.std(axis=0).mean()
            var_norm = _norm(var, 0, 5.0)  # 5 以上认为活跃
        else:
            var_norm = 0.5

        # repeat ratio: 最近 8 步里独立位置比例的倒数
        recent = list(self.pos_history)[-8:]
        uniq = len(set(recent))
        repeat_ratio = 1.0 - (uniq / max(len(recent), 1))

        # revisit: 最近 1 步是否到访过旧格子
        revisit = 1.0 if (self.cur_pos in self.visited_set and self.step_no > 1) else 0.0
        revisit_count_norm = _norm(len(self.visited_set), 0, 500)  # 访问的格子越多越好

        # bounce: 动作执行了但位置没动
        bounce = 0.0
        if self.last_action >= 0 and self.prev_pos == self.cur_pos:
            bounce = 1.0
        self.bounce_flag = bounce

        # 综合 stuck 判定(不是特征, 是奖励用)
        self.stuck_flag = 1.0 if (repeat_ratio > 0.6 and var_norm < 0.2) else 0.0

        return np.array([var_norm, repeat_ratio, revisit_count_norm, bounce], dtype=np.float32)

    def _build_env_meta_feat(self):
        return np.array(
            [
                self.battery_max / 999.0,
                self.max_step / 2000.0,
                self.npc_count_meta / 4.0,
                self.station_count_meta / 10.0,
                self.charger_count_meta / 4.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # 实体 slot                                                           #
    # ------------------------------------------------------------------ #
    def _relative(self, target_pos):
        dx = target_pos[0] - self.cur_pos[0]
        dy = target_pos[1] - self.cur_pos[1]
        dist = float(np.sqrt(dx * dx + dy * dy))
        return dx, dy, dist

    def _build_station_feat(self):
        target_ids = set(self.packages)
        # 排序: 目标优先, 然后按距离
        def key(s):
            cid = s.get("config_id", 0)
            dx, dy, d = self._relative((s["pos"]["x"], s["pos"]["z"]))
            return (0 if cid in target_ids else 1, d)

        sorted_st = sorted(self.stations, key=key)[: Config.K_STATION]

        feats = []
        for i in range(Config.K_STATION):
            if i < len(sorted_st):
                s = sorted_st[i]
                dx, dy, d = self._relative((s["pos"]["x"], s["pos"]["z"]))
                is_tgt = 1.0 if s.get("config_id", 0) in target_ids else 0.0
                cfg_id = s.get("config_id", 0)
                feats.extend(
                    [
                        1.0,
                        _sym_norm(dx, MAP_COORD_SCALE),
                        _sym_norm(dy, MAP_COORD_SCALE),
                        _sym_norm(s["pos"]["x"], MAP_COORD_SCALE),
                        _sym_norm(s["pos"]["z"], MAP_COORD_SCALE),
                        _norm(d, 0, MAX_DIST),
                        is_tgt,
                        _norm(cfg_id, 0, 10),
                    ]
                )
            else:
                feats.extend([0.0] * Config.STATION_FEAT)
        return np.array(feats, dtype=np.float32)

    def _build_npc_feat(self):
        # 1) 当前帧按距离升序 (最近的威胁最重要)
        npcs_with_dist = [
            (n, self._relative((n["pos"]["x"], n["pos"]["z"]))) for n in self.npcs
        ]
        npcs_with_dist.sort(key=lambda t: t[1][2])
        npcs_sorted = npcs_with_dist[: Config.K_NPC]

        # 2) 针对当前每个 NPC, 用"上一帧最近位置"做最近邻匹配, 得到 approach
        #    approach_rate = (prev_dist_to_self - cur_dist_to_self) / NORM
        #    NPC 一步位移最多 ~2 格, 所以归一化 scale 取 2.
        APPROACH_NORM = 2.0

        # 对当前每只 NPC, 在 prev_npc_positions 里找离它空间最近的上帧位置,
        # 用"上帧位置 -> 当前我方位置"的距离作为 prev_dist_to_self.
        # 这样即使 NPC 每帧移动 1~2 格也能稳定对上.
        # (NPC <= 4 个, O(K^2) 完全可以接受)
        def _match_prev_dist_to_self(cur_npc_pos):
            if not self.prev_npc_positions:
                return None
            best_match = None
            best_sq = None
            for px, pz in self.prev_npc_positions:
                sq = (px - cur_npc_pos[0]) ** 2 + (pz - cur_npc_pos[1]) ** 2
                if best_sq is None or sq < best_sq:
                    best_sq = sq
                    best_match = (px, pz)
            if best_match is None:
                return None
            # 上一帧这只 NPC 到"当前"我方位置的距离
            # (用当前位置近似, 因为我自己一步只走 1~1.4 格, 误差可接受)
            dx = best_match[0] - self.cur_pos[0]
            dz = best_match[1] - self.cur_pos[1]
            return float(np.sqrt(dx * dx + dz * dz))

        feats = []
        self.cur_npc_approach_rate = 0.0
        self.cur_npc_threat = 0.0
        for i in range(Config.K_NPC):
            if i < len(npcs_sorted):
                n, (dx, dy, d) = npcs_sorted[i]
                # 威胁: 半径 15 格 + 平方衰减, 更陡峭 (旧: 线性, 半径 10)
                threat = float(np.clip(1.0 - d / 15.0, 0.0, 1.0))
                threat = threat * threat
                # approach
                prev_d = _match_prev_dist_to_self((n["pos"]["x"], n["pos"]["z"]))
                if prev_d is None:
                    approach = 0.0
                else:
                    approach = float(np.clip((prev_d - d) / APPROACH_NORM, -1.0, 1.0))
                # 给奖励用: 记录最近那只 NPC 的 approach/threat
                if i == 0:
                    self.cur_npc_approach_rate = approach
                    self.cur_npc_threat = threat

                feats.extend(
                    [
                        1.0,
                        _sym_norm(dx, MAP_COORD_SCALE),
                        _sym_norm(dy, MAP_COORD_SCALE),
                        _norm(d, 0, MAX_DIST),
                        threat,
                        approach,
                    ]
                )
            else:
                feats.extend([0.0] * Config.NPC_FEAT)

        # 更新"上一帧"缓存: 保留当前这一帧所有可见 NPC 的绝对位置
        self.prev_npc_positions = [
            (n["pos"]["x"], n["pos"]["z"]) for n in self.npcs
        ]
        return np.array(feats, dtype=np.float32)

    def _build_charger_feat(self):
        chargers_sorted = sorted(
            self.chargers,
            key=lambda c: self._relative((c["pos"]["x"], c["pos"]["z"]))[2],
        )[: Config.K_CHARGER]

        batt_ratio = self.battery / max(self.battery_max, 1)
        need_charge = 1.0 if batt_ratio < 0.35 else 0.0

        feats = []
        for i in range(Config.K_CHARGER):
            if i < len(chargers_sorted):
                c = chargers_sorted[i]
                dx, dy, d = self._relative((c["pos"]["x"], c["pos"]["z"]))
                # priority: 距离 / 电量 — 距离越近 + 电量越低, 优先级越高
                # 归一化到 [0,1]
                priority = float(np.clip((1.0 - d / MAX_DIST) * (1.0 - batt_ratio), 0, 1))
                feats.extend(
                    [
                        1.0,
                        _sym_norm(dx, MAP_COORD_SCALE),
                        _sym_norm(dy, MAP_COORD_SCALE),
                        _norm(d, 0, MAX_DIST),
                        need_charge,
                        priority,
                    ]
                )
            else:
                feats.extend([0.0] * Config.CHARGER_FEAT)
        return np.array(feats, dtype=np.float32)

    def _build_warehouse_feat(self):
        w_sorted = sorted(
            self.warehouses,
            key=lambda w: self._relative((w["pos"]["x"], w["pos"]["z"]))[2],
        )[: Config.K_WAREHOUSE]

        need_resupply = 1.0 if len(self.packages) == 0 else 0.0

        feats = []
        for i in range(Config.K_WAREHOUSE):
            if i < len(w_sorted):
                w = w_sorted[i]
                dx, dy, d = self._relative((w["pos"]["x"], w["pos"]["z"]))
                feats.extend(
                    [
                        1.0,
                        _sym_norm(dx, MAP_COORD_SCALE),
                        _sym_norm(dy, MAP_COORD_SCALE),
                        _norm(d, 0, MAX_DIST),
                        need_resupply,
                        0.0,
                    ]
                )
            else:
                feats.extend([0.0] * Config.WAREHOUSE_FEAT)
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 局部地图: 21x21 可通行网格 -> 7x7 (3x3 max-pool)                      #
    # ------------------------------------------------------------------ #
    def _build_local_map_feat(self):
        if self.map_info is None:
            return np.zeros(Config.LOCAL_MAP_DIM, dtype=np.float32)
        try:
            m = np.array(self.map_info, dtype=np.float32)
            if m.shape != (21, 21):
                # 不是 21x21 就填 0, 避免异常
                return np.zeros(Config.LOCAL_MAP_DIM, dtype=np.float32)
            # 3x3 max-pool 得到 7x7
            m7 = m.reshape(7, 3, 7, 3).max(axis=(1, 3))
            return m7.flatten().astype(np.float32)
        except Exception:
            return np.zeros(Config.LOCAL_MAP_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 【action_aux 阶段新增】动作级辅助信号 (27 维) - 直击避碰 NPC       #
    #   [0:8]   ACTION_RISK    每个动作方向 5 步 look-ahead 的 NPC 追击风险
    #   [8:16]  ACTION_BLOCK   每个动作方向 +1/+2 格是否被 NPC 占位 (0/1)
    #   [16:24] NPC_VELOCITY   最多 4 个 NPC 的 (vx, vy) 速度, 按 dist 升序
    #   [24:27] NEAREST_POLAR  最近 NPC 的 (sin_angle, cos_angle, TTC)
    # 设计要点:
    #   - 动作级信号直接告诉网络"这个方向安不安全", 免去小 MLP 的反推压力
    #   - NPC 速度让网络区分"朝我来"与"横向走"的 NPC, 是 approach 的补充
    # ------------------------------------------------------------------ #
    def _build_action_aux_feat(self):
        K = Config.K_NPC
        # ---------- 1. 收集可用的 NPC 信息 (最多 K 个, 按距离升序) ----------
        npc_list = []  # list of dict: {pos, vel, dist_self}
        cur = self.cur_pos
        prev_map = self._build_prev_npc_map_for_aux()  # 独立缓存, 跨帧有效

        for n in self.npcs:
            nx, nz = n["pos"]["x"], n["pos"]["z"]
            d = float(np.sqrt((nx - cur[0]) ** 2 + (nz - cur[1]) ** 2))
            prev_pos = prev_map.get((nx, nz))
            if prev_pos is None:
                vx, vz = 0.0, 0.0
            else:
                vx = float(nx - prev_pos[0])
                vz = float(nz - prev_pos[1])
            npc_list.append({"nx": nx, "nz": nz, "vx": vx, "vz": vz, "d": d})
        npc_list.sort(key=lambda t: t["d"])
        npc_list = npc_list[:K]

        # ---------- 2. ACTION_RISK (8 维) ----------
        # 对每个动作方向, 模拟向前走 t 步 (t=1..5), NPC 按当前速度走 t 步,
        # 若彼此距离 <= 1.5 判定会碰, risk = (1 - t/5) * threat(d_now).
        # 取所有 NPC 中最大 risk. 若无 NPC 威胁则 0.
        LOOKAHEAD = 5
        CONTACT_RADIUS = 1.5
        risk = np.zeros(8, dtype=np.float32)
        for a in range(8):
            dx_a, dy_a = ACTION_DELTA[a]
            max_risk = 0.0
            for n in npc_list:
                # threat 归一化: 距离 15 格外没威胁
                if n["d"] > 15.0:
                    continue
                threat = (1.0 - n["d"] / 15.0) ** 2
                for t in range(1, LOOKAHEAD + 1):
                    mx = cur[0] + dx_a * t
                    mz = cur[1] + dy_a * t
                    nx_t = n["nx"] + n["vx"] * t
                    nz_t = n["nz"] + n["vz"] * t
                    dd = np.sqrt((mx - nx_t) ** 2 + (mz - nz_t) ** 2)
                    if dd <= CONTACT_RADIUS:
                        risk_t = float((1.0 - (t - 1) / LOOKAHEAD) * threat)
                        if risk_t > max_risk:
                            max_risk = risk_t
                        break  # 这个 NPC 沿本动作最早的碰撞点锁定
            risk[a] = max_risk

        # ---------- 3. ACTION_BLOCK (8 维) ----------
        # 对每个动作方向 +1/+2 格落点, 如果有 NPC 在曼哈顿半径 <= 1 内, 置 1.
        block = np.zeros(8, dtype=np.float32)
        for a in range(8):
            dx_a, dy_a = ACTION_DELTA[a]
            blocked = False
            for steps in (1, 2):
                px = cur[0] + dx_a * steps
                pz = cur[1] + dy_a * steps
                for n in npc_list:
                    if abs(n["nx"] - px) <= 1 and abs(n["nz"] - pz) <= 1:
                        blocked = True
                        break
                if blocked:
                    break
            block[a] = 1.0 if blocked else 0.0

        # ---------- 4. NPC_VELOCITY (8 维) ----------
        # 最多 K=4 个 NPC 的 (vx, vy), 归一化到 [-1, 1] (一步最多 1~2 格)
        VEL_NORM = 2.0
        vel = np.zeros(Config.NPC_VELOCITY_DIM, dtype=np.float32)
        for i, n in enumerate(npc_list):
            vel[i * 2 + 0] = float(np.clip(n["vx"] / VEL_NORM, -1.0, 1.0))
            vel[i * 2 + 1] = float(np.clip(n["vz"] / VEL_NORM, -1.0, 1.0))

        # ---------- 5. NEAREST_POLAR (3 维) ----------
        # (sin_angle, cos_angle, TTC_norm)
        # angle = atan2(dy, dx); TTC = d / max(closing_speed, ε); 归一化到 [0, 1]
        polar = np.zeros(Config.NEAREST_NPC_POLAR_DIM, dtype=np.float32)
        if npc_list:
            n = npc_list[0]
            dx_r = n["nx"] - cur[0]
            dz_r = n["nz"] - cur[1]
            angle = float(np.arctan2(dz_r, dx_r))
            polar[0] = float(np.sin(angle))
            polar[1] = float(np.cos(angle))
            # 闭合速度 = - (NPC 径向相对速度); 径向速度 = 速度向量 · 单位方向(NPC->我)
            # 等价于: closing = dot(-v_npc, unit_dir_npc_to_self)
            if n["d"] > 1e-3:
                ux = -dx_r / n["d"]
                uz = -dz_r / n["d"]
                closing = n["vx"] * (-ux) + n["vz"] * (-uz)   # 朝我来时 closing > 0
            else:
                closing = 2.0
            if closing > 0.05:
                ttc = n["d"] / closing
                polar[2] = float(np.clip(1.0 - ttc / 10.0, 0.0, 1.0))
            else:
                polar[2] = 0.0

        return np.concatenate([risk, block, vel, polar]).astype(np.float32)

    def _build_prev_npc_map_for_aux(self):
        """把 prev_npc_positions_for_aux (上帧坐标集合) 映射到当前帧最近的
        NPC 位置. 返回 dict: (cur_nx, cur_nz) -> (prev_px, prev_pz).
        只有当 prev 里能找到距离 <=3 格的最近邻时才算匹配 (避免跨 NPC 错配).

        注意: 这个函数**同时**会在返回前刷新 self.prev_npc_positions_for_aux
        为当前帧 NPC 位置, 用于下次调用. 独立于 _build_npc_feat 里的
        prev_npc_positions, 避免两者互相覆盖.
        """
        result = {}
        if self.prev_npc_positions_for_aux and self.npcs:
            used = set()
            for n in self.npcs:
                cx, cz = n["pos"]["x"], n["pos"]["z"]
                best = None
                best_sq = 9.01   # 距离 >3 就不认
                for i, (px, pz) in enumerate(self.prev_npc_positions_for_aux):
                    if i in used:
                        continue
                    sq = (px - cx) ** 2 + (pz - cz) ** 2
                    if sq < best_sq:
                        best_sq = sq
                        best = (i, px, pz)
                if best is not None:
                    used.add(best[0])
                    result[(cx, cz)] = (best[1], best[2])
        # 刷新 for_aux 缓存 (下一帧会用)
        self.prev_npc_positions_for_aux = [
            (n["pos"]["x"], n["pos"]["z"]) for n in self.npcs
        ]
        return result

    # ------------------------------------------------------------------ #
    # 合法动作 (NPC-aware mask, 距离判定版)                                #
    # ------------------------------------------------------------------ #
    # 关键发现: 环境的碰撞判定包含"相邻格"和"对角相邻格" (距离 <= sqrt(2) 即碰撞).
    # 简单的"位置完全重叠" mask 会漏掉相邻/对角相邻的危险, 所以用几何距离判定:
    #   对每个 hero 动作 a, 若 hero_next(a) 到某个 NPC 可达位置 的距离 <= DANGER_RADIUS,
    #   则 mask 掉 a.
    # 两层兜底防止"全 mask":
    #   兜底1: 全 mask → 收紧只对距离 hero <= CRITICAL_DIST 的 NPC 做 mask
    #   兜底2: 仍全 mask → 回到 base (让网络自己抉择)
    NPC_MASK_RADIUS_HERO = 4.0      # hero 距离 NPC <= 此值时才触发 mask
    NPC_MASK_CRITICAL_DIST = 1.5    # CRITICAL 层: 兜底时只考虑这些极近 NPC
    NPC_DANGER_RADIUS = 1.5         # hero_next 与 NPC_next 的"危险距离": 对角相邻=√2, 留点余量

    def _build_legal_action(self):
        # 1. 环境原始 legal_action
        la = [int(x) for x in (self.legal_act or [1] * 8)][:8]
        while len(la) < 8:
            la.append(1)
        if sum(la) == 0:
            la = [1] * 8
        base = np.array(la, dtype=np.int32)

        # 2. 找出需要避让的 NPC (精确保存坐标)
        close_npcs = []        # dist <= NPC_MASK_RADIUS_HERO
        critical_npcs = []     # dist <= NPC_MASK_CRITICAL_DIST (兜底专用)
        for n in self.npcs:
            npx = int(n.get("pos", {}).get("x", 0))
            npy = int(n.get("pos", {}).get("z", 0))
            dx = npx - self.cur_pos[0]
            dy = npy - self.cur_pos[1]
            d = float(np.sqrt(dx * dx + dy * dy))
            if d <= self.NPC_MASK_RADIUS_HERO:
                close_npcs.append((npx, npy))
            if d <= self.NPC_MASK_CRITICAL_DIST:
                critical_npcs.append((npx, npy))

        if not close_npcs:
            return base.tolist()

        # 3. 生成 NPC 可达点列表 (每个 NPC: 不动 + 8 方向 = 9 点)
        def _reach_points(npc_list):
            pts = []
            for (npx, npy) in npc_list:
                pts.append((npx, npy))
                for dx, dy in ACTION_DELTA:
                    pts.append((npx + int(dx), npy + int(dy)))
            return pts

        npc_reach = _reach_points(close_npcs)

        # 4. 距离判定: 对每个动作, 若 hero_next 到任一 NPC 可达点 <= DANGER_RADIUS, mask
        danger_r2 = self.NPC_DANGER_RADIUS * self.NPC_DANGER_RADIUS  # 避免开方
        def _make_mask(reach_pts):
            m = np.array(base, copy=True)
            for a in range(Config.ACTION_NUM):
                if m[a] == 0:
                    continue
                ddx, ddy = ACTION_DELTA[a]
                hx = self.cur_pos[0] + int(ddx)
                hy = self.cur_pos[1] + int(ddy)
                for (rx, ry) in reach_pts:
                    dx2 = (hx - rx) ** 2 + (hy - ry) ** 2
                    if dx2 <= danger_r2:
                        m[a] = 0
                        break
            return m

        safe = _make_mask(npc_reach)

        # 5. 兜底 1: 全 mask → 只考虑 CRITICAL NPC (最近的)
        if safe.sum() == 0:
            if critical_npcs:
                safe = _make_mask(_reach_points(critical_npcs))
            else:
                safe = base.copy()

        # 6. 兜底 2: 仍全 mask → 完全回退
        if safe.sum() == 0:
            safe = base.copy()

        return safe.tolist()

    # ------------------------------------------------------------------ #
    # 奖励计算 — 续训阶段 2                                                #
    # ------------------------------------------------------------------ #
    def _compute_reward(self, is_new_pos: bool):
        w = Config.RewardWeights
        r = 0.0

        batt_ratio = self.battery / max(self.battery_max, 1)
        LOW_BATT = Config.LOW_BATTERY_THRESHOLD
        NPC_DANGER = Config.NPC_DANGER_DIST

        # ---------- 正向主信号 ---------- #
        # 1. 投递奖励 (主目标)
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            r += w.DELIVERY * newly_delivered

            # 1a. 连投加成: 本次距上次投递 <= window 步 → 鼓励"多单一趟打"
            gap = self.step_no - self.last_delivery_step
            if gap <= Config.DELIVERY_COMBO_WINDOW:
                r += w.DELIVERY_COMBO * newly_delivered

            # 1b. 剩余电量奖励: 投递时电量还充足 → 鼓励"高效走位"而非绕远
            if batt_ratio >= Config.BATT_LEFT_DELIVERY_BAR:
                r += w.BATT_LEFT_AT_DELIVERY * newly_delivered

            self.last_delivery_step = self.step_no

        # 2. 取货(回仓库)
        if self.back_warehouse_count > self.last_back_warehouse_count:
            r += w.PICKUP

        # 3. 充电(低电量时奖励, 高电量时不奖励避免反复进充电桩)
        if self.charge_count > self.last_charge_count:
            if batt_ratio < 0.5:
                r += w.CHARGE * (1.0 - batt_ratio)

        # ---------- NPC 负向 + 主动避让奖励 ---------- #
        npc_dist = self._nearest_npc_dist()
        self.last_npc_dist = npc_dist
        if npc_dist is not None:
            if self.ep_min_npc_dist is None or npc_dist < self.ep_min_npc_dist:
                self.ep_min_npc_dist = npc_dist
            if npc_dist <= 1.5:
                r += w.NPC_CRITICAL
            elif npc_dist <= NPC_DANGER:
                r += w.NPC_NEAR

        # 5b. 连续 approach 信号 (加强): 逼近 & 威胁大 -> 扣分
        if self.cur_npc_approach_rate > 0.0 and self.cur_npc_threat > 0.0:
            r += w.NPC_APPROACH * self.cur_npc_approach_rate * self.cur_npc_threat

        # 5c. 【续训 v2 新增】EVADE_MOVE — 危险时段内成功拉开距离时奖励
        # 条件: 上帧 & 当前帧都在危险时段 (npc_dist <= 3),
        #       且当前帧最近 NPC 距离 > 上帧 -> 说明这步"躲对了".
        # 奖励量级: EVADE_MOVE * (Δd clip 到 [0, 2])
        in_danger_now = (npc_dist is not None and npc_dist <= NPC_DANGER)
        in_danger_prev = (self.prev_npc_dist is not None and self.prev_npc_dist <= NPC_DANGER)
        if in_danger_now and in_danger_prev and npc_dist > self.prev_npc_dist:
            r += w.EVADE_MOVE * float(np.clip(npc_dist - self.prev_npc_dist, 0.0, 2.0))
        self.prev_npc_dist = npc_dist

        # ---------- 势函数 (目标 + 低电保命) ---------- #
        # 4a. 向最近目标驿站靠近 (危险时段关闭, 让策略专心躲)
        target_dist = self._nearest_target_dist()
        if target_dist is not None and self.prev_target_dist is not None and not in_danger_now:
            delta = self.prev_target_dist - target_dist
            r += w.POTENTIAL_TO_TARGET * float(np.clip(delta, -2.0, 2.0))
        self.prev_target_dist = target_dist

        # 4b. 【续训 v2 新增】低电量时向最近充电桩靠近 — 保命势能
        # 触发条件: batt_ratio < LOW_BATT, 且不在 NPC 危险时段(避免冲突信号)
        charger_dist = self._nearest_charger_dist()
        if batt_ratio < LOW_BATT and charger_dist is not None and self.prev_charger_dist is not None and not in_danger_now:
            delta_c = self.prev_charger_dist - charger_dist
            r += w.POTENTIAL_TO_CHARGER * float(np.clip(delta_c, -2.0, 2.0))
        self.prev_charger_dist = charger_dist

        # ---------- 低电惩罚 (阈值放宽) ---------- #
        # 阈值 0.20 -> 0.30: 提前预警. 但单点惩罚从 -0.01 -> -0.006 (conf 已改),
        # 保持"总量"接近但覆盖面更广.
        if batt_ratio < LOW_BATT:
            self.ep_low_batt_steps += 1
            if charger_dist is None or charger_dist > 6:
                r += w.LOW_BATTERY_NO_CHARGE

        # ---------- 卡角 / bounce / novelty / step ---------- #
        if self.stuck_flag > 0:
            self.ep_stuck_count += 1
            r += w.STUCK
        if self.bounce_flag > 0:
            self.ep_bounce_count += 1
            r += w.BOUNCE

        if is_new_pos:
            r += w.NOVELTY

        r += w.STEP

        return [float(r)]

    def _nearest_target_dist(self):
        if not self.packages or not self.stations:
            return None
        tset = set(self.packages)
        ds = [
            self._relative((s["pos"]["x"], s["pos"]["z"]))[2]
            for s in self.stations if s.get("config_id", 0) in tset
        ]
        return min(ds) if ds else None

    def _nearest_npc_dist(self):
        if not self.npcs:
            return None
        ds = [self._relative((n["pos"]["x"], n["pos"]["z"]))[2] for n in self.npcs]
        return min(ds) if ds else None

    def _nearest_charger_dist(self):
        if not self.chargers:
            return None
        ds = [self._relative((c["pos"]["x"], c["pos"]["z"]))[2] for c in self.chargers]
        return min(ds) if ds else None
