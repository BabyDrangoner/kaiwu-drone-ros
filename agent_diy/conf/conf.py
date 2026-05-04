#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 2026  DIY Author
###########################################################################
"""
DIY Agent config for Drone Delivery.

核心设计原则:
1. 所有归一化量尽量使用"比例/无量纲"形式, 让 battery_max / max_step 变化时
   网络无需重学.
2. 实体(驿站 / NPC / 充电桩)采用"定长 slot + found 标志位"格式,
   K 固定为设计上限, 局内实际数量少则 padding 0.
3. 网络按实体类型分组编码, 再用简单 attention / masked pooling 聚合,
   避免实体数量变动时特征错位.
4. 网络容量克制, 目标 <= 80K 参数, 适合 i7 CPU 本地训练.
"""


class Config:
    # ---------------- 动作 & 价值 ---------------- #
    ACTION_NUM = 8
    VALUE_NUM = 1
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    NUMB_HEAD = 1

    # ---------------- 实体 slot 数 ---------------- #
    # 按环境允许的最大值取, 训练时实际少于这个值也能正常工作
    K_STATION = 10   # 驿站最多 10 (station_count 范围 3~10)
    K_NPC = 4        # NPC 最多 4 (drone_count 范围 1~4)
    K_CHARGER = 4    # 充电桩最多 4 (charger_count 范围 1~4)
    K_WAREHOUSE = 1  # 仓库 1 个

    # 每个实体 slot 的特征维度
    STATION_FEAT = 8   # [found, dx, dy, ax, ay, dist, is_target, target_idx_norm]
    NPC_FEAT = 6       # [found, dx, dy, dist, threat, approach]
    CHARGER_FEAT = 6   # [found, dx, dy, dist, need_charge, priority]
    WAREHOUSE_FEAT = 6 # [found, dx, dy, dist, need_resupply, pad]

    # ---------------- 自身状态 ---------------- #
    # [batt_ratio, batt_low_flag, pkg_ratio, has_pkg, step_ratio,
    #  pos_x, pos_y, delivered_norm, charge_count_norm,
    #  back_warehouse_norm, action_hist_4x8=32 ...]
    SELF_STATE_DIM = 10
    ACTION_HISTORY_LEN = 4  # 最近 4 步动作 one-hot
    ACTION_HIST_DIM = ACTION_HISTORY_LEN * ACTION_NUM  # 4*8 = 32

    # 卡角落相关指示器 / stuck 检测
    STUCK_DIM = 4  # [pos_variance_norm, repeat_ratio, revisit_count_norm, bounce_flag]

    # ---------------- 局部视野(21x21) ---------------- #
    # 保留一份压缩后的障碍物分布, 用简单 MLP 编码而不是 CNN(CPU 友好)
    # 把 21x21 按 3x3 max-pool 下采样成 7x7 = 49 维
    LOCAL_MAP_SIZE = 7
    LOCAL_MAP_DIM = LOCAL_MAP_SIZE * LOCAL_MAP_SIZE  # 49

    # ---------------- 全局环境常量(每局固定, 用于网络自适应) ---------------- #
    # 这些值随局而变, 我们把它们的"归一化后形式"也作为特征, 让网络知道"本局难度"
    ENV_META_DIM = 5  # [battery_max/999, max_step/2000, npc_cnt/4, station_cnt/10, charger_cnt/4]

    # ---------------- 动作辅助信号 (v1+action_aux 阶段新增) ---------------- #
    # 专门为"避碰 NPC"设计的 action-level 信号, 拼接在所有老特征之后.
    # 目的: 直接告诉网络"选每个方向的安全度", 不再让小 MLP 去反推.
    # 维度细分:
    #   ACTION_RISK:   8 维  每个动作方向的"5 步内被 NPC 追上"风险分数
    #   ACTION_BLOCK:  8 维  每个动作方向 +1/+2 格内是否有 NPC (0/1)
    #   NPC_VELOCITY:  8 维  最多 4 个 NPC 的 (vx, vy) 速度向量
    #   NEAREST_NPC_POLAR: 3 维 最近 NPC 的 (sin_angle, cos_angle, TTC)
    ACTION_RISK_DIM = 8
    ACTION_BLOCK_DIM = 8
    NPC_VELOCITY_DIM = K_NPC * 2         # = 8
    NEAREST_NPC_POLAR_DIM = 3
    ACTION_AUX_DIM = (
        ACTION_RISK_DIM
        + ACTION_BLOCK_DIM
        + NPC_VELOCITY_DIM
        + NEAREST_NPC_POLAR_DIM
    )  # 27

    # ---------------- 拼接总维度 ---------------- #
    # self(10) + action_hist(32) + stuck(4) + env_meta(5) = 51
    # station(8*10=80) + npc(6*4=24) + charger(6*4=24) + warehouse(6*1=6) = 134
    # local_map(49) = 49
    # action_aux(27) = 27
    # total = 51 + 134 + 49 + 27 = 261
    STATION_TOTAL = STATION_FEAT * K_STATION            # 80
    NPC_TOTAL = NPC_FEAT * K_NPC                        # 24
    CHARGER_TOTAL = CHARGER_FEAT * K_CHARGER            # 24
    WAREHOUSE_TOTAL = WAREHOUSE_FEAT * K_WAREHOUSE      # 6
    SELF_TOTAL = SELF_STATE_DIM + ACTION_HIST_DIM + STUCK_DIM + ENV_META_DIM  # 51

    FEATURE_LEN = (
        SELF_TOTAL
        + STATION_TOTAL
        + NPC_TOTAL
        + CHARGER_TOTAL
        + WAREHOUSE_TOTAL
        + LOCAL_MAP_DIM
        + ACTION_AUX_DIM
    )  # 261
    DIM_OF_OBSERVATION = FEATURE_LEN

    # 各段的 offset, 供网络分组读取
    OFFSET_SELF = 0
    OFFSET_STATION = SELF_TOTAL
    OFFSET_NPC = OFFSET_STATION + STATION_TOTAL
    OFFSET_CHARGER = OFFSET_NPC + NPC_TOTAL
    OFFSET_WAREHOUSE = OFFSET_CHARGER + CHARGER_TOTAL
    OFFSET_MAP = OFFSET_WAREHOUSE + WAREHOUSE_TOTAL
    OFFSET_ACTION_AUX = OFFSET_MAP + LOCAL_MAP_DIM  # = 234, 老特征尾部

    # ---------------- PPO / 训练超参 ---------------- #
    # 【续训阶段 2 — from step=788792】
    # 策略已跑满 ~14h, 再次进一步降低 lr / entropy / clip, 让奖励变化
    # 引导策略做精细优化, 不被噪声冲坏.
    GAMMA = 0.995
    LAMDA = 0.95
    START_LR = 5e-5                      # 1e-4 -> 5e-5, 更细粒度调参
    INIT_LEARNING_RATE_START = START_LR
    CLIP_PARAM = 0.12                    # 0.15 -> 0.12, 更保守
    VF_COEF = 0.5
    VALUE_LOSS_COEFF = VF_COEF
    BETA_START = 0.003                   # 0.005 -> 0.003, 策略已成熟
    ENTROPY_LOSS_COEFF = BETA_START
    GRAD_CLIP_RANGE = 0.5
    USE_GRAD_CLIP = True

    # 价值头归一化: 奖励量级约 (-1, +1) 区间, 无需额外缩放

    # ---------------- 网络 ---------------- #
    HIDDEN_DIM = 64           # 主干 hidden
    ENTITY_EMBED_DIM = 32     # 每类实体编码后维度
    MAP_EMBED_DIM = 32        # 局部地图编码后维度
    ACTION_AUX_EMBED_DIM = 32 # 动作辅助信号编码后维度

    # ---------------- 奖励权重 ---------------- #
    # 【续训阶段 2】核心思想: 不再靠堆负向惩罚, 而是为每个痛点配一个
    # "做对了能赚回来"的正向信号, 避免策略缩头不敢动.
    #   - NPC 碰撞:  NPC_APPROACH 稍加强 + 新增 EVADE_MOVE (躲开获得奖励)
    #   - 电量归零:  提前预警阈值 20% -> 30%, 新增朝桩势函数 + 低电惩罚下调
    #   - 提分:      DELIVERY 1.0 -> 1.5, 新增 DELIVERY_COMBO 连投加成,
    #                新增 BATT_LEFT_AT_DELIVERY 投递时电量高额外奖励
    # 每步最多净收益 ~+0.14 (理想躲避), 净损失 ~-0.19 (严重危险),
    # 投递一次净 ~+1.9, 主信号占主导.
    class RewardWeights:
        # 正向 — 主信号
        DELIVERY = 1.5                   # 1.0 -> 1.5, 拉大主信号
        DELIVERY_COMBO = 0.3             # 【新增】距上次投递 <= 80 步时额外加
        BATT_LEFT_AT_DELIVERY = 0.1      # 【新增】投递时电量 > 50% 额外加
        PICKUP = 0.05
        CHARGE = 0.03                    # 低电进桩时 × (1 - batt_ratio)
        POTENTIAL_TO_TARGET = 0.02       # 向目标驿站靠近 (危险时段会被关闭)
        POTENTIAL_TO_CHARGER = 0.02      # 【新增】低电(<30%)时向最近桩靠近
        EVADE_MOVE = 0.02                # 【新增】危险时段(d<=3)拉开 NPC 距离
        NOVELTY = 0.005
        # 负向
        STEP = -0.0005
        BATTERY_DRAIN = -0.0
        LOW_BATTERY_NO_CHARGE = -0.006   # -0.01 -> -0.006, 阈值放宽但减小单点
        NPC_NEAR = -0.1                  # 与最近 NPC 距离 <=3 格
        NPC_CRITICAL = -0.5              # 与最近 NPC 距离 <=1.5 格
        NPC_APPROACH = -0.08             # -0.05 -> -0.08, 连续逼近稍加强
        STUCK = -0.01
        BOUNCE = -0.02
        TERMINATED_FAIL = -3.0           # 局末异常终止, 已经够强, 保持
        TRUNCATED_NODELIVER = -0.3

    # ---------------- 奖励阈值 / 窗口常量 ---------------- #
    # 【续训阶段 2】把硬编码阈值提到 class 级别, 方便后续统一调.
    LOW_BATTERY_THRESHOLD = 0.30         # 0.20 -> 0.30, 提前预警
    DELIVERY_COMBO_WINDOW = 80           # 两次投递间隔 <=80 步才算 combo
    BATT_LEFT_DELIVERY_BAR = 0.50        # 投递时电量 > 50% 获得额外奖励
    NPC_DANGER_DIST = 3.0                # 危险时段判定: 最近 NPC <=3 格
